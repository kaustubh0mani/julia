# This file is a part of Julia. License is MIT: http://julialang.org/license

using Base.Iterators.Enumerate

"""
    asyncmap(f, c...; ntasks=0, batch_size=nothing) -> collection

Uses multiple concurrent tasks to map `f` over a collection (or multiple
equal length collections). For multiple collection arguments, `f` is
applied elementwise.

`ntasks` specifies the number of tasks to run concurrently.
If unspecified, upto a 100 tasks will be started to process the collection
(depends on its length).

`ntasks` can also be specified as a zero-arg function. In this case, the
number of tasks to start is checked before processing every element and a new
task started if the value of `ntasks_func()` is less than the current number
of tasks.

If `batch_size` is specified, the collection is processed in batch mode. `f` should
then be a function that should accept a `Vector{Any}` of argument tuples and should
return a vector of results. The input vector will have a length upto `batch_size`

The following examples return the `object_id` of the tasks in which the mapping
function is executed for different invocations of `ntasks` and `batch_size`.

With ntasks undefined, each element is processed in a different task.
```
julia> tskoid() = object_id(current_task());

julia> asyncmap(x->tskoid(), 1:5)
5-element Array{UInt64,1}:
 0x6e15e66c75c75853
 0x440f8819a1baa682
 0x9fb3eeadd0c83985
 0xebd3e35fe90d4050
 0x29efc93edce2b961

julia> length(unique(asyncmap(x->tskoid(), 1:5)))
5
```

With `ntasks=2` all elements are only processed in 2 tasks.
```
julia> asyncmap(x->tskoid(), 1:5; ntasks=2)
5-element Array{UInt64,1}:
 0x027ab1680df7ae94
 0xa23d2f80cd7cf157
 0x027ab1680df7ae94
 0xa23d2f80cd7cf157
 0x027ab1680df7ae94

julia> length(unique(asyncmap(x->tskoid(), 1:5; ntasks=2)))
2
```

With `batch_size` defined, the mapping function needs to be changed to accept an array
of argument tuples. And return an array of results. In this example we call `map`
on the batch input to achieve the same.
```
julia> batch_func(input) = map(x->string("args_tuple: ", x, ", element_val: ", x[1], ", task: ", tskoid()), input)
batch_func (generic function with 1 method)

julia> asyncmap(batch_func, 1:5; ntasks=2, batch_size=2)
5-element Array{String,1}:
 "args_tuple: (1,), element_val: 1, task: 9118321258196414413"
 "args_tuple: (2,), element_val: 2, task: 4904288162898683522"
 "args_tuple: (3,), element_val: 3, task: 9118321258196414413"
 "args_tuple: (4,), element_val: 4, task: 4904288162898683522"
 "args_tuple: (5,), element_val: 5, task: 9118321258196414413"
```

!!! note
    Currently, all tasks in Julia are executed in a single OS thread co-operatively. Consequently,
    `ayncmap` is beneficial only when the mapping function involves any I/O - disk, network, remote
    worker invocation, etc.

"""
function asyncmap(f, c...; ntasks=0, batch_size=nothing)
    ntasks = verify_ntasks(c[1], ntasks)

    if isa(batch_size, Number)
        @assert batch_size >= 0
        exec_func = batch -> begin
            # extract the Refs from the input tuple
            batch_refs = map(x->x[1], batch)

            # and the args tuple....
            batched_args = map(x->x[2], batch)

            results = f(batched_args)
            foreach(x -> (batch_refs[x[1]].x = x[2]), enumerate(results))
        end
    else
        exec_func = (r,args) -> (r.x = f(args...))
    end
    chnl, worker_tasks = setup_chnl_and_tasks(exec_func, ntasks, batch_size)
    return wrap_n_exec_twice(chnl, worker_tasks, ntasks, exec_func, c...)
end

function verify_ntasks(iterable, ntasks)
    if ntasks == 0
        chklen = iteratorsize(iterable)
        if (chklen == HasLength()) || (chklen == HasShape())
            ntasks = max(1,min(100, length(iterable)))
        else
            ntasks = 100
        end
    end
    ntasks
end

function wrap_n_exec_twice(chnl, worker_tasks, ntasks, exec_func, c...)
    # The driver task, creates a Ref object and writes it and the args tuple to
    # the communication channel for processing by a free worker task.
    push_arg_to_channel = (x...) -> (r=Ref{Any}(nothing); put!(chnl,(r,x));r)

    if isa(ntasks, Function)
        map_f = (x...) -> begin
            # check number of tasks everytime, and start one if required.
            # number_tasks > optimal_number is fine, the other way around is inefficient.
            if length(worker_tasks) < ntasks()
                start_worker_task!(worker_tasks, exec_func, chnl)
            end
            push_arg_to_channel(x...)
        end
    else
        map_f = push_arg_to_channel
    end
    maptwice(map_f, chnl, worker_tasks, c...)
end

function maptwice(wrapped_f, chnl, worker_tasks, c...)
    # first run, returns a collection of Refs
    asyncrun_excp = nothing
    local asyncrun
    try
        asyncrun = map(wrapped_f, c...)
    catch ex
        if isa(ex,InvalidStateException)
            # channel could be closed due to exceptions in the async tasks,
            # we should propagate those errors, if any, over the `put!` failing
            # in asyncrun.
            asyncrun_excp = ex
        else
            rethrow(ex)
        end
    end

    # close channel and wait for all worker tasks to finish
    close(chnl)

    # check and throw any exceptions from the worker tasks
    foreach(x->(v=wait(x); isa(v, Exception) && throw(v)), worker_tasks)

    # check if there was a genuine problem with asyncrun
    (asyncrun_excp != nothing) && throw(asyncrun_excp)

    # second run, extract values from the Refs and return
    map(x->x.x, asyncrun)
end

function setup_chnl_and_tasks(exec_func, ntasks, batch_size=nothing)
    if isa(ntasks, Function)
        nt = ntasks()
        # start at least one worker task.
        if nt == 0
            nt=1
        end
    else
        nt = ntasks
    end

    # Use an unbuffered channel for communicating with the worker tasks. In the event
    # of any error in any of the worker tasks, the channel is closed. And this
    # results in the `put!` in the driver task immediately failing.
    chnl=Channel(0)
    worker_tasks = []
    foreach(_ -> start_worker_task!(worker_tasks, exec_func, chnl, batch_size), 1:nt)

    return (chnl, worker_tasks)
end

function start_worker_task!(worker_tasks, exec_func, chnl, batch_size=nothing)
    t = @schedule begin
        retval = nothing

        try
            if isa(batch_size, Number)
                while isopen(chnl)
                    # The mapping function expects an array of input args, as it processes
                    # elements in a batch.
                    batch_collection=Any[]
                    n=0
                    for exec_data in chnl
                        push!(batch_collection, exec_data)
                        n += 1
                        (n == batch_size) && break
                    end
                    if n > 0
                        exec_func(batch_collection)
                    end
                end
            else
                for exec_data in chnl
                    exec_func(exec_data...)
                end
            end
        catch e
            close(chnl)
            retval = e
        end
        retval
    end
    push!(worker_tasks, t)
end


"""
    AsyncCollector(f, results, c...; ntasks=0) -> iterator

Apply `f` to each element of `c` using at most `ntasks` asynchronous
tasks.

If `ntasks` is unspecified, upto a 100 tasks may be used.
For multiple collection arguments, apply `f` elementwise.
Output is collected into `results`.

!!! note
    `next(::AsyncCollector, state) -> (nothing, state)`
!!! note
    `for task in AsyncCollector(f, results, c...) end` is equivalent to
    `map!(f, results, c...)`.
"""
type AsyncCollector
    f
    results
    enumerator::Enumerate
    ntasks
    nt_check::Bool     # check number of tasks on every iteration

    AsyncCollector(f, r, en::Enumerate, ntasks) = new(f, r, en, ntasks, isa(ntasks, Function))
end

function AsyncCollector(f, results, c...; ntasks=0)
    AsyncCollector(f, results, enumerate(zip(c...)), ntasks)
end

type AsyncCollectorState
    chnl::Channel
    worker_tasks::Array{Task,1}
    enum_state      # enumerator state
end

function start(itr::AsyncCollector)
    itr.ntasks = verify_ntasks(itr.enumerator, itr.ntasks)
    chnl, worker_tasks = setup_chnl_and_tasks((i,args) -> (itr.results[i]=itr.f(args...)) , itr.ntasks)
    return AsyncCollectorState(chnl, worker_tasks, start(itr.enumerator))
end


function done(itr::AsyncCollector, state::AsyncCollectorState)
    if !isopen(state.chnl) || done(itr.enumerator, state.enum_state)
        close(state.chnl)

        # wait for all tasks to finish
        foreach(x->(v=wait(x); isa(v, Exception) && throw(v)), state.worker_tasks)
        empty!(state.worker_tasks)
        return true
     else
        return false
     end
end

function next(itr::AsyncCollector, state::AsyncCollectorState)
    if itr.nt_check && (length(state.worker_tasks) < itr.ntasks())
        start_worker_task!(state.worker_tasks, itr.f, state.chnl)
    end

    # Get index and mapped function arguments from enumeration iterator.
    (i, args), state.enum_state = next(itr.enumerator, state.enum_state)
    put!(state.chnl, (i, args))

    return (nothing, state)
end

"""
    AsyncGenerator(f, c...; ntasks=0) -> iterator

Apply `f` to each element of `c` using at most `ntasks` asynchronous tasks.
If `ntasks` is unspecified, upto a 100 tasks may be used.
For multiple collection arguments, apply `f` elementwise.
Results are returned by the iterator as they become available.

!!! note
    `collect(AsyncGenerator(f, c...; ntasks=1))` is equivalent to
    `map(f, c...)`.
"""
type AsyncGenerator
    collector::AsyncCollector
end

function AsyncGenerator(f, c...; ntasks=0)
    AsyncGenerator(AsyncCollector(f, Dict{Int,Any}(), c...; ntasks=ntasks))
end

type AsyncGeneratorState
    i::Int
    collector_state::AsyncCollectorState
end

start(itr::AsyncGenerator) = AsyncGeneratorState(0, start(itr.collector))

# Done when source async collector is done and all results have been consumed.
function done(itr::AsyncGenerator, state::AsyncGeneratorState)
    done(itr.collector, state.collector_state) && isempty(itr.collector.results)
end

function next(itr::AsyncGenerator, state::AsyncGeneratorState)
    state.i += 1

    results_dict = itr.collector.results
    while !haskey(results_dict, state.i)
        if done(itr.collector, state.collector_state)
            # `done` waits for async tasks to finish. if we do not have the index
            # we are looking for, it is an error.
            !haskey(results_dict, state.i) && error("Error processing index ", i)
            break;
        end
        _, state.collector_state = next(itr.collector, state.collector_state)
    end
    r = results_dict[state.i]
    delete!(results_dict, state.i)

    return (r, state)
end

# pass-through iterator traits to the iterable
# on which the mapping function is being applied
iteratorsize(itr::AsyncGenerator) = iteratorsize(itr.collector.enumerator)
size(itr::AsyncGenerator) = size(itr.collector.enumerator)
length(itr::AsyncGenerator) = length(itr.collector.enumerator)

"""
    asyncmap!(f, c)

In-place version of [`asyncmap()`](:func:`asyncmap`).
"""
asyncmap!(f, c) = (for x in AsyncCollector(f, c, c) end; c)


"""
    asyncmap!(f, results, c...)

Like [`asyncmap()`](:func:`asyncmap`), but stores output in `results` rather than
returning a collection.
"""
asyncmap!(f, r, c1, c...) = (for x in AsyncCollector(f, r, c1, c...) end; r)
