#define DOCTEST_CONFIG_IMPLEMENT
#include <doctest/doctest.h>

#include <tbb/global_control.h>

int main(int argc, char **argv) {
    tbb::task_scheduler_handle handle{oneapi::tbb::attach{}};
    int retcode = doctest::Context(argc, argv).run();
    // explicitly finalizate TBB worker threads
    // without this, valgrind detects active TBB allocations
    tbb::finalize(handle);
    return retcode;
}
