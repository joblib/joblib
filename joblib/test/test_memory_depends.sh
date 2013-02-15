rm -rf  testdepends

echo '1st launch'
python test_memory_depends.py | more

echo 'Wait to launch 2nd launch. All must be read from cache'
read test
python test_memory_depends.py | more

echo 'Now modify the dependency. Then observe the cache is invalidated'
read test
python test_memory_depends.py | more

