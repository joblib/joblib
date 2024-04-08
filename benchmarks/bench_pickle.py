"""
Benching joblib pickle I/O.

Warning: this is slow, and the benches are easily offset by other disk
activity.
"""
import os
import time
import shutil
import numpy as np
import joblib
import gc

from joblib.disk import disk_used

try:
    from memory_profiler import memory_usage
except ImportError:
    memory_usage = None


def clear_out():
    """Clear output directory."""
    if os.path.exists('out'):
        shutil.rmtree('out')
    os.mkdir('out')


def kill_disk_cache():
    """Clear disk cache to avoid side effects."""
    if os.name == 'posix' and os.uname()[0] == 'Linux':
        try:
            os.system('sudo sh -c "sync; echo 3 > /proc/sys/vm/drop_caches"')
        except IOError as e:
            if e.errno == 13:
                print('Please run me as root')
            else:
                raise
    else:
        # Write ~100M to the disk
        open('tmp', 'wb').write(np.random.random(2e7))


def delete_obj(obj):
    """Force destruction of an object."""
    if obj is not None:
        del obj
    gc.collect()


def memory_used(func, *args, **kwargs):
    """Compute memory usage of func."""
    if memory_usage is None:
        return np.nan

    gc.collect()
    mem_use = memory_usage((func, args, kwargs), interval=.001)

    return max(mem_use) - min(mem_use)


def timeit(func, *args, **kwargs):
    """Compute the mean execution time of func based on 7 measures."""
    times = []
    tries = kwargs['tries']
    kwargs.pop('tries')
    if tries > 1:
        tries += 2

    for _ in range(tries):
        kill_disk_cache()
        t0 = time.time()
        out = func(*args, **kwargs)
        if 1:
            # Just time the function
            t1 = time.time()
            times.append(t1 - t0)
        else:
            # Compute a hash of the output, to estimate the time
            # necessary to access the elements: this is a better
            # estimate of the time to load with me mmapping.
            joblib.hash(out)
            t1 = time.time()
            joblib.hash(out)
            t2 = time.time()
            times.append(t2 - t0 - 2 * (t2 - t1))
    times.sort()
    return np.mean(times[1:-1]) if tries > 1 else t1 - t0, out


def generate_rand_dict(size,
                       with_arrays=False,
                       with_string=False,
                       array_shape=(10, 10)):
    """Generate dictionary with random values from list of keys."""
    ret = {}
    rnd = np.random.RandomState(0)
    randoms = rnd.random_sample((size))
    for key, random in zip(range(size), randoms):
        if with_arrays:
            ret[str(key)] = rnd.random_sample(array_shape)
        elif with_string:
            ret[str(key)] = str(random)
        else:
            ret[str(key)] = int(random)
    return ret


def generate_rand_list(size,
                       with_arrays=False,
                       with_string=False,
                       array_shape=(10, 10)):
    """Generate list with random values from list of keys."""
    ret = []
    rnd = np.random.RandomState(0)
    for random in rnd.random_sample((size)):
        if with_arrays:
            ret.append(rnd.random_sample(array_shape))
        elif with_string:
            ret.append(str(random))
        else:
            ret.append(int(random))
    return ret


def print_line(dataset, strategy,
               write_time, read_time,
               mem_write, mem_read,
               disk_used):
    """Nice printing function."""
    print('% 15s, %12s, % 6.3f, % 7.4f, % 9.1f, % 9.1f, % 5.1f' % (
          dataset, strategy,
          write_time, read_time,
          mem_write, mem_read, disk_used))


def print_bench_summary(args):
    """Nice bench summary function."""
    summary = """Benchmark summary:
    - Global values:
        . Joblib version: {}
        . Number of tries to compute mean execution time: {}
        . Compression levels   : {}
        . Compression algorithm: {}
        . Memory map mode      : {}
        . Bench nifti data     : {}
        . Bench big array      : {}
        . Bench 2 big arrays   : {}
        . Bench big dictionary: {}
        . Bench array+dict     : {}
""".format(joblib.__version__,
           args.tries,
           ", ".join(map(str, args.compress)),
           "None" if not args.compress else args.compressor,
           args.mmap,
           args.nifti,
           args.array,
           args.arrays,
           args.dict,
           args.combo)

    if args.array:
        shape = tuple(args.shape)
        size = round(np.multiply.reduce(shape) * 8 / 1024 ** 2, 1)
        summary += """
    - Big array:
        . shape: {}
        . size in memory: {} MB
""".format(str(shape), size)

    if args.dict:
        summary += """
    - Big dictionary:
        . number of keys: {}
        . value type: {}
""".format(args.size, 'np.ndarray'
           if args.valuearray else 'str'
            if args.valuestring else 'int')
        if args.valuearray:
            summary += """        . arrays shape: {}
""".format(str(tuple(args.valuearrayshape)))

    if args.list:
        summary += """
    - Big list:
        . number of elements: {}
        . value type: {}
""".format(args.size, 'np.ndarray'
           if args.valuearray else 'str'
            if args.valuestring else 'int')
        if args.valuearray:
            summary += """        . arrays shape: {}
""".format(str(tuple(args.valuearrayshape)))

    print(summary)


def bench_compress(dataset, name='',
                   compress=('zlib', 0), cache_size=0, tries=5):
    """Bench joblib dump and load functions, compress modes."""
    # generate output compression strategy string before joblib compatibility
    # check as it may override the compress variable with a non tuple type.
    compress_str = "Raw" if compress[1] == 0 else "{} {}".format(*compress)

    # joblib versions prior to 0.10 doesn't support tuple in compress argument
    # so only the second element of the tuple is used for those versions
    # and the compression strategy is ignored.
    if (isinstance(compress, tuple) and
            tuple(map(int, joblib.__version__.split('.')[:2])) < (0, 10)):
        compress = compress[1]

    time_write = time_read = du = mem_read = mem_write = []
    clear_out()
    time_write, obj = timeit(joblib.dump, dataset, 'out/test.pkl',
                             tries=tries,
                             compress=compress, cache_size=cache_size)
    del obj
    gc.collect()
    mem_write = memory_used(joblib.dump, dataset, 'out/test.pkl',
                            compress=compress, cache_size=cache_size)
    delete_obj(dataset)
    du = disk_used('out') / 1024.
    time_read, obj = timeit(joblib.load, 'out/test.pkl', tries=tries)
    delete_obj(obj)
    mem_read = memory_used(joblib.load, 'out/test.pkl')
    print_line(name, compress_str, time_write, time_read,
               mem_write, mem_read, du)


def bench_mmap(dataset, name='', cache_size=0, mmap_mode='r', tries=5):
    """Bench joblib dump and load functions, memmap modes."""
    time_write = time_read = du = []
    clear_out()
    time_write, _ = timeit(joblib.dump, dataset, 'out/test.pkl',
                           tries=tries,
                           cache_size=cache_size)
    mem_write = memory_used(joblib.dump, dataset, 'out/test.pkl',
                            cache_size=cache_size)

    delete_obj(dataset)

    time_read, obj = timeit(joblib.load, 'out/test.pkl',
                            tries=tries,
                            mmap_mode=mmap_mode)
    delete_obj(obj)
    mem_read = memory_used(joblib.load, 'out/test.pkl', mmap_mode=mmap_mode)
    du = disk_used('out') / 1024.
    print_line(name, 'mmap %s' % mmap_mode,
               time_write, time_read, mem_write, mem_read, du)


def run_bench(func, obj, name, **kwargs):
    """Run the benchmark function."""
    func(obj, name, **kwargs)


def run(args):
    """Run the full bench suite."""
    if args.summary:
        print_bench_summary(args)

    if (not args.nifti and not args.array and not args.arrays and
            not args.dict and not args.list and not args.combo):
        print("Nothing to bench. Exiting")
        return

    compress_levels = args.compress
    compress_method = args.compressor
    mmap_mode = args.mmap

    container_size = args.size
    a1_shape = tuple(args.shape)
    a2_shape = (10000000, )

    print('% 15s, %12s, % 6s, % 7s, % 9s, % 9s, % 5s' % (
          'Dataset', 'strategy', 'write', 'read',
          'mem_write', 'mem_read', 'disk'))

    if args.nifti:
        # Nifti images
        try:
            import nibabel
        except ImportError:
            print("nibabel is not installed skipping nifti file benchmark.")
        else:
            def load_nii(filename):
                img = nibabel.load(filename)
                return img.get_data(), img.get_affine()

            for name, nifti_file in (
                    ('MNI',
                     '/usr/share/fsl/data/atlases'
                     '/MNI/MNI-prob-1mm.nii.gz'),
                    ('Juelich',
                     '/usr/share/fsl/data/atlases'
                     '/Juelich/Juelich-prob-2mm.nii.gz'), ):
                for c_order in (True, False):
                    name_d = '% 5s(%s)' % (name, 'C' if c_order else 'F')
                    for compress_level in compress_levels:
                        d = load_nii(nifti_file)

                        if c_order:
                            d = (np.ascontiguousarray(d[0]), d[1])

                        run_bench(bench_compress, d, name_d,
                                  compress=(compress_method, compress_level),
                                  tries=args.tries)
                        del d
                    if not args.nommap:
                        d = load_nii(nifti_file)
                        if c_order:
                            d = (np.ascontiguousarray(d[0]), d[1])

                        run_bench(bench_mmap, d, name_d,
                                  mmap_mode=mmap_mode, tries=args.tries)
                        del d

    # Generate random seed
    rnd = np.random.RandomState(0)

    if args.array:
        # numpy array
        name = '% 5s' % 'Big array'
        for compress_level in compress_levels:
            a1 = rnd.random_sample(a1_shape)
            run_bench(bench_compress, a1, name,
                      compress=(compress_method, compress_level),
                      tries=args.tries)
            del a1
        if not args.nommap:
            a1 = rnd.random_sample(a1_shape)
            run_bench(bench_mmap, a1, name, mmap_mode=mmap_mode,
                      tries=args.tries)
            del a1

    if args.arrays:
        # Complex object with 2 big arrays
        name = '% 5s' % '2 big arrays'
        for compress_level in compress_levels:
            obj = [rnd.random_sample(a1_shape), rnd.random_sample(a2_shape)]
            run_bench(bench_compress, obj, name,
                      compress=(compress_method, compress_level),
                      tries=args.tries)
            del obj
        if not args.nommap:
            obj = [rnd.random_sample(a1_shape), rnd.random_sample(a2_shape)]
            run_bench(bench_mmap, obj, name, mmap_mode=mmap_mode,
                      tries=args.tries)
            del obj

    if args.dict:
        # Big dictionary
        name = '% 5s' % 'Big dict'
        array_shape = tuple(args.valuearrayshape)
        for compress_level in compress_levels:
            big_dict = generate_rand_dict(container_size,
                                          with_arrays=args.valuearray,
                                          with_string=args.valuestring,
                                          array_shape=array_shape)
            run_bench(bench_compress, big_dict, name,
                      compress=(compress_method, compress_level),
                      tries=args.tries)
            del big_dict
        if not args.nommap:
            big_dict = generate_rand_dict(container_size,
                                          with_arrays=args.valuearray,
                                          with_string=args.valuestring,
                                          array_shape=array_shape)
            run_bench(bench_mmap, big_dict, name, mmap_mode=mmap_mode,
                      tries=args.tries)
            del big_dict

    if args.list:
        # Big dictionary
        name = '% 5s' % 'Big list'
        array_shape = tuple(args.valuearrayshape)
        for compress_level in compress_levels:
            big_list = generate_rand_list(container_size,
                                          with_arrays=args.valuearray,
                                          with_string=args.valuestring,
                                          array_shape=array_shape)
            run_bench(bench_compress, big_list, name,
                      compress=(compress_method, compress_level),
                      tries=args.tries)
            del big_list
        if not args.nommap:
            big_list = generate_rand_list(container_size,
                                          with_arrays=args.valuearray,
                                          with_string=args.valuestring,
                                          array_shape=array_shape)
            run_bench(bench_mmap, big_list, name, mmap_mode=mmap_mode,
                      tries=args.tries)
            del big_list

    if args.combo:
        # 2 big arrays with one big dict
        name = '% 5s' % 'Dict/arrays'
        array_shape = tuple(args.valuearrayshape)
        for compress_level in compress_levels:
            obj = [rnd.random_sample(a1_shape),
                   generate_rand_dict(container_size,
                                      with_arrays=args.valuearray,
                                      with_string=args.valuestring,
                                      array_shape=array_shape),
                   rnd.random_sample(a2_shape)]
            run_bench(bench_compress, obj, name,
                      compress=(compress_method, compress_level),
                      tries=args.tries)
            del obj
        if not args.nommap:
            obj = [rnd.random_sample(a1_shape),
                   generate_rand_dict(container_size,
                                      with_arrays=args.valuearray,
                                      with_string=args.valuestring,
                                      array_shape=array_shape),
                   rnd.random_sample(a2_shape)]
            run_bench(bench_mmap, obj, name,
                      mmap_mode=mmap_mode,
                      tries=args.tries)
            del obj


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Joblib benchmark script")
    parser.add_argument('--compress', nargs='+', type=int, default=(0, 3),
                        help="List of compress levels.")
    parser.add_argument('--compressor', type=str, default='zlib',
                        choices=['zlib', 'gzip', 'bz2', 'xz', 'lzma'],
                        help="Compression algorithm.")
    parser.add_argument('--mmap', type=str, default='r',
                        choices=['r', 'r+', 'w+'],
                        help="Memory map mode.")
    parser.add_argument('--tries', type=int, default=5,
                        help="Number of tries to compute execution time"
                             "mean on.")
    parser.add_argument('--shape', nargs='+', type=int, default=(10000, 10000),
                        help="Big array shape.")
    parser.add_argument("-m", "--nommap", action="store_true",
                        help="Don't bench memmap")
    parser.add_argument('--size', type=int, default=10000,
                        help="Big dictionary size.")
    parser.add_argument('--valuearray', action="store_true",
                        help="Use numpy arrays type in containers "
                             "(list, dict)")
    parser.add_argument('--valuearrayshape', nargs='+', type=int,
                        default=(10, 10),
                        help="Shape of arrays in big containers.")
    parser.add_argument('--valuestring', action="store_true",
                        help="Use string type in containers (list, dict).")
    parser.add_argument("-n", "--nifti", action="store_true",
                        help="Benchmark Nifti data")
    parser.add_argument("-a", "--array", action="store_true",
                        help="Benchmark single big numpy array")
    parser.add_argument("-A", "--arrays", action="store_true",
                        help="Benchmark list of big numpy arrays")
    parser.add_argument("-d", "--dict", action="store_true",
                        help="Benchmark big dictionary.")
    parser.add_argument("-l", "--list", action="store_true",
                        help="Benchmark big list.")
    parser.add_argument("-c", "--combo", action="store_true",
                        help="Benchmark big dictionary + list of "
                             "big numpy arrays.")
    parser.add_argument("-s", "--summary", action="store_true",
                        help="Show bench summary.")

    run(parser.parse_args())
