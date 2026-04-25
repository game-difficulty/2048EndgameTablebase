#include <iostream>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            std::cout << "Max threads: " << omp_get_max_threads() << std::endl;
        }
    }
    return 0;
}
