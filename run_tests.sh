function test {
	rm -rf build/test
	mkdir -p build/test
	cd build
	cmake -DCMAKE_INSTALL_PREFIX=local -DCMAKE_CXX_FLAGS="$@" ..
	make install
	cd ..

	for i in test/*.cpp; do
		echo compiling $i
		g++ $i -o build/$i.test -lgtest -Lbuild/local/lib -Ibuild/local/include -llightflow
		echo running $i
		LD_LIBRARY_PATH=build/local/lib build/$i.test
	done
}

set -e
rm -rf build
echo "$@"
test "$@"