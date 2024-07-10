
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>
#include <random>

#include <iostream>
#include <fstream>

struct MarkovChainFunctor {
public:

    MarkovChainFunctor( Kokkos::Random_XorShift64_Pool<Kokkos::CudaSpace> pool_) : 
        pool(pool_) {}

    KOKKOS_INLINE_FUNCTION
    void operator()(const int i, double& sum) const {
        auto generator = pool.get_state();

        double random_num = generator.drand(0.0, 2.0);
        sum += random_num;

        pool.free_state(generator);
    }
        
private:
    Kokkos::Random_XorShift64_Pool<Kokkos::CudaSpace> pool;
};

int main(int argc, char** argv){

    Kokkos::initialize(argc, argv);

    {
        std::random_device rd;
        std::uniform_int_distribution<int> dist(0, 4000);

        Kokkos::Random_XorShift64_Pool<Kokkos::CudaSpace> random_pool(dist(rd));
        Kokkos::DefaultExecutionSpace().print_configuration(std::cout);

        double p = 811193170652452;
        double sum = 0;

        // TODO handle the case where p is too large to fit into an int
        // We can use long long int, but we may need to consider 
        // external batching to the GPU as well
        int64_t dim = static_cast<int64_t>(p);

        std::cout << std::numeric_limits<int64_t>::max() << std::endl;

        std::cout << dim << std::endl;

        Kokkos::RangePolicy<Kokkos::IndexType<int64_t>> policy(0, dim);

        Kokkos::parallel_reduce("MarkovChainReduce", policy, MarkovChainFunctor(random_pool), sum);

        Kokkos::fence();

        std::cout << sum / p << std::endl;
    }

    Kokkos::finalize();
}
