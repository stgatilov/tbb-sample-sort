I started this project after I tried to port [In-place Parallel Super Scalar Samplesort (IPS⁴o)](https://github.com/ips4o/ips4o) to TBB task system and Visual C++.
The IPS4o uses parallelization model of N simultaneously running threads synchronizing with each other using barriers, which is hard to properly port to a task system.
While I made some messy prototype using `task::suspend`, the final abomination looked too scary to rely on.

This project is in fact a rewrite of [PS⁴o](https://github.com/ips4o/ps4o).
It follows IPS4o algorithm in general but uses the temporary second array of elements.
Since the algorithm is no longer in-place, the implementation is much simpler, and it can naturally run on TBB.
At the cost of being somewhat slower of course =)

On Ryzen 5 9600x, sorting 200 millions of 64-bit integers takes:
* 720 ms for IPS4o
* 870 ms for PS4o and TBBSS


### Extra memory

Two large temporary buffers are used, of sizes `N` and `N * sizeof(T)` bytes respectively.
Using the second array of elements has many performance drawbacks:
 1. Working set is 2x more, more physical RAM is required on the same problem size to avoid going into swap.
 2. Before allocation, the OS has to zero the memory pages it maps into the virtual address space.
 3. During deallocation, the OS has to crawl through all the unmapped memory pages and update the page table entries.
 4. When elements are moved from one large array to the other, CPU has to read the cache lines at the destination array unnecessarily.
 5. About 50% of the elements end up in the wrong array and have to be copied back into the user's array in the end.

Issue 4 (read from destination) can be solved by using SSE2 uncached writes.
It is controlled by `TBBSS_UNCACHED_MININPUT_BYTES` macro and is enabled by default on x86.
Quite surprisingly, I don't notice much difference in the wall time on my machine if I disable the macro.

The issue 2 (zeroing allocated pages) can be hard to notice in a benchmark because most of the OSes can zero memory pages ahead of time as well as zero them lazily on the first use.
But it affects the overall performance of the application nevertheless.

The issue 3 (page table update on free) is very well noticeable.
On my machine, it adds 5-10% to the wall time, which is spent inside kernel call on a single thread.
It can't be parallelized because operations on virtual address space are protected by a mutex, and for the same reason it is can hard to overlap with other useful computations.
It seems that huge pages can reduce this time almost to zero, but they don't always work and have too many drawbacks to consider them seriously in general.
Trasparent Huge Pages can be enabled with macro `TBBSS_LARGE_PAGES`.


### Element lifetime

In my opinion, fast sorting algorithms only need to work on small elements of primitive and on simple struct types.
If you want to sort `std::string`-s as fast as possible, either just use `tbb::parallel_sort` or search for specialized algorithms.
However, this library provides some ways to sort elements with non-trivial lifetime management.

The algorithm keeps logically dead elements as uninitialized memory, and thus never uses default constructor of the user type.
Usually it relocates (destructing move) elements around, but it also copies a few elements sometimes.
Most of the types are trivially relocatable, and the library can work with such types in a special way.

This is described in detail in enum `RelocationTrivialness` and value traits.
Roughly speaking, in the safest mode `rtNone` the algorithm invokes all the correct methods: move constructor/assignment, destructor, copy constructor.
In the fastest mode `rtFork` the algorithm only copies bytes around with memcpy or similar functions, none of the standard C++ lifetime methods are ever called.
While such handling of C++ types most likely invokes undefined behavior, it has many benefits, including the ability to use branchless algorithms for small size sort.


### Exceptions and cancellation

There are two possible sources of exceptions inside the algorithm:
 1. Memory allocation routines throw std::bad_alloc if allocations fails.
 2. Element lifetime methods can throw. This is especially likely for copy constructors.

I made quite some effort to properly handle the first type of exceptions, and I believe the algorithm works as expected if some memory allocation throws.
The alive elements spread across various locations are all relocated back into the user storage by RAII guards, so you'll see the N input elements in undefined order in the sorted array outside.
There is even a unit test checking this.

The second case is **not** handled properly: `std::terminate` sill happen if lifetime method throws.
If your elements are that picky, you can either force `rtFork` mode or just use `tbb::parallel_sort`.

The cancellation without exception through TBB is handled properly too, there is also a unit test on it.


### Algorithm

This algorithm is a comparison sort, it only extracts information about elements by comparing them to each other.
The algorithm is not stable: equal elements can be reordered in arbitrary way.
However, the algorithm is deterministic: across separate runs and different platforms equal elements are reordered equally.
And the same set of comparisons, moves, and copies can be observed.
Non-cryptographic pseudorandom generator is used to select samples and pivots.

The samplesort algorithm (also called multi-pivot quicksort) is a divide-and-conquer algorithm.
To sort a segment of elements, it selects (B - 1) pivots and rearranges the elements into B <= 256 buckets.
Then each bucket is sorted recursively.
Small buckets are processed using simple quadratic algorithm.
To select pivots, some number of random samples is taken and sorted recursively.
The highest-level multi-pivot partition is fully parallelized.
To avoid bad performance on many equal elements, all the elements equal to a duplicate pivot are put into an individual bucket and excluded from recursion.

The algorithm relies a lot on branchless processing.
The multi-pivot partition first classifies input elements into buckets.
Following PS4o, the classification phase runs branchless binary search on heap-ordered tree of pivots.
Small sort algorithm uses branchless compare-and-swap to minimize branching (unless you use `rtNone` mode), see `TBBSS_BRANCHLESS_COMPARESWAP`.
