To my surprise I wasn't able to find an easy to understand example of MapReduce implementation in python. Re-creating the pattern from paper turned out to be a fairly nuanced endeavour. Lets peel this onion together.

# Bite-sized MapReduce in python

This article is an introductory practical material for educational purposes. Folks that plan to actually work with Hadoop are advised to trust none of it.

## The algorithm

High level idea is fairly straight forward. Given a set of files containing records we first call map(record) to produce pairs of (key, value). Then all pairs of the same key are directed to a `reduce(key, values[])` call output of which is written into a partial output file.

Taking a closer look, semantics of both methods becomes slightly more peculiar:
`map(record: Object) -> List[Tuple[key, value0]]`
`reduce(key, values: Iterable) -> List[Tuple[key, value1]]`
i.e. single map may produce multiple pairs out of a single record; reduce may produce a list of outputs but generally expected to have 0 or 1.

## Lingvo

Lets pin down the names of different portions of data, actors, and processes:
- input_split - one of 1..N files processed by MapReduce. Single input_split becomes a single map assignment
- shuffle_block - a part of map output transferred between mapper_i and reducer_j.
- part_file - output of a single reducer and subsequently output of the algorithm

Shuffle is routing of data between map and reduce. Partitioning by key, transferring those partitions, and preparing reducer input by merge/sort/group. Essentially, everything that happens between return from a `map()` and start of `reduce()` call.

Coordinator (also prehistorically known as master) is the object that coordinates execution of MapReduce across multiple Mappers and Reducers.

Mapper is an actor that is being assigned to run map on one or multiple input_splits. Mapper also is responsible for splitting it's own output into shuffle_blocks (one per input_split per reducer) and pre-sorting the pairs by key within each split.

Reducer is one of possibly multiple actors that runs reduce() function. Reducer accumulates shuffle_blocks, generates groups of `(key, values[])` for the reduce call, and writes the output data.

DFS (distributed file system) is a piece of infrastructure that handles all data reads/writes across distributed actors. Note that Coordinator is not expected to interact with DFS whatsoever.

## Interactions:

Here we start to put it all together into a functional system. I may be making a number of arbitrary choices regarding details of some interactions that would differ from a real world system. For example, original publication states that Coordinator notifies Reducers of data readiness, while in Hadoop Reducers discover it themselves.

![MapReduce interactions](/images/mapReduce.png)

1. User:
   1.1 -> Write input_splits to DFS
   1.2 -> Call mapReduce(job) on Coordinator
   1.3 <- Await done from Coordinator
   1.4 -> Read part_files from DFS
2. Coordinator:
   2.1 -> Assign map task(input_split) to Mapper
   2.2 <- Await shuffle_blocks ready from Mapper
   2.3 -> Notify Reducer of shuffle_block ready
   2.4 <- Await all map completions
   2.5 -> Send ready_to_reduce to Reducer
   2.6 <- Await reduce complete from Reducer
   2.7 -> Notify User done
3. Mapper:
   3.1 <- Await map task assignment from Coordinator
   3.2 -> Read input_split from DFS
   3.3 [] Produce [0..N] pairs per record; pre-sort; combine
   3.4 -> Write shuffle_block to DFS
   3.5 -> Notify Coordinator shuffle_blocks ready
4. Reducer:
   4.1 <- Await shuffle_block ready from Coordinator
   4.2 -> Copy shuffle_block from DFS
   4.3 <- Await ready_to_reduce from Coordinator
   4.4 [] Iterate over groups; call reduce()
   4.5 -> Write part_file to DFS
   4.6 -> Notify Coordinator reduce complete
5. DFS:
   - Store/serve input_splits
   - Store/serve shuffle_blocks
   - Store part_files

## Deep dives

Algorithm inputs and outputs. Input is a list of file locations that can be used directly as a mapper assignment. Similar, output of the algorithm is also a list of files, one per reducer. Output list is ordered by the reducer that produced and it is expected that if partitioning logic implied an order this order will be preserved in output files. For example, if we had a range partition across 2 reducers such that the 1st handles letters [A-K] and the other [L-Z], the first file is expected to be produced with keys [A-K] and second [L-Z]: `[part_file_0_AK, part_file_1_LZ]`.

Sorting vs partitioning. Key can be a compound object `<fname, lname, dob>`. Default partitioning schema resorts to `hash(key) % R` but can have a custom configuration of `hash(key["fname"])` for example. Partition function operates on the key only and never on the full record.

Reducer mechanics. Reducer fetches the shuffle_blocks as they become ready, but the reduce itself is only called once all the blocks are ready. Reducer can iterate over N pre-sorted blocks in a "Merge k-sorted lists" manner, therefore avoiding ever materializing the complete partition in memory.

Mapper can emit [0..N] intermediate pairs from a single record.


## Advanced topics (not covered)

1. Fault tolerance. If one of the workers fails it's task has to be reassigned to a different worker. Task may need to be redone if the worker was also part of DFS.
2. Filtering intermediate results.
3. Temporary splitting of *hot* keys. If key distribution is heavily skewed it may cause an overload on a reducer. It requires additional logic to split frequent keys.
4. Combine. In order to reduce amount of data being transfered between mapper and reducer it is sometimes possible to do a local partial reduce on mapper node. 

## Implementation details.

We organize code around actors and interactions described in section ##Interactions.

Parallel workers (Map, Reduce) are configured with an internal ThreadExecutionPool to run the payload tasks.

Messaging between Coordinator and Workers is organized using pairs of Queues. Each queue message is declared as a class for readability:
- `MapMsg` — Coordinator → Mapper: map task assignment
- `MapCompleteMsg` — Mapper → Coordinator: shuffle_blocks ready
- `ShuffleBlockReadyMsg` — Coordinator → Reducer: shuffle_block location
- `ReduceMsg` — Coordinator → Reducer: ready to reduce
- `ReduceCompleteMsg` — Reducer → Coordinator: part_file location


## Code

Complete implementation code is available at https://gist.github.com/yselivonchyk/1c2cc91f9402bb578d221b8266c2b8a3