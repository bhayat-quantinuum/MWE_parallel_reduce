# MWE_parallel_reduce
A temporary repo to demonstrate an issue I am facing with Kokkos

Kokkos was built from commit `4f416f3b7056b14fc5f9367fbdd77e3cede205bc` of the Kokkos repo.

Try varying the value of `p` in the code to observe different behaviour. 

For small `p` everything works fine.

For large `p` (I have tried with 811193170652452) the parallel reduce doesn't execute at all and we silently fail.
