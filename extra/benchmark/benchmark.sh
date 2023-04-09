function bench {
    export LD_LIBRARY_PATH=../../build/local/lib
    g++ -llightflow -DLF_CUDA_AVAIL benchmark.cc -o lfbench
    echo $1
    echo "-- CPU --"
    export LF_DEFDEV=0
    ./lfbench > lfres
    python benchmark.py
    echo "-- CUDA --"
    export LF_DEFDEV=1
    ./lfbench > lfres
    python benchmark.py
    rm lfres lfbench
}

mkdir -p runs
bench $1 | tee runs/$(date +"%y%m%d-%H%M%S")
