function bench {
    g++ -llightflow benchmark.cc -o lfbench
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
bench | tee runs/$(date +"%y%m%d-%H%M%S")
