g++ -llightflow benchmark.cc -o lfbench
echo "benchmarking lightflow ..."
./lfbench > lfres
echo "benchmarking pyversion ..."
taskset -c 0 python benchmark.py
rm lfres lfbench