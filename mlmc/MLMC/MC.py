import numpy as np
from mlmc.xp import xp

import typer
from typing import Tuple

from .Factory import IntegratorFactory

import mpi4py.MPI
import mpi4py.futures

import threading # To call wait on asynchronous sends...
from mlmc.ParallelExecution.ParallelExecutor import ParallelExecutor

import time

app = typer.Typer()

# A class for MC simulations of the process
class MC:
    def __init__(self,
    dt: float,
    n     : int,
    x0: xp.ndarray,
    i_f: IntegratorFactory,
    phi) -> None:
        """
        beta: the beta variable
        dt: the time step
        n:  the number of walkers
        x0: the initial condition
        sf: the scatter factory to obtain the potential etc...
        phi:    the function to estimate
        """
        self.dt = dt
        self.n = n
        self.x0 = x0
        self.i_f = i_f
        self.phi = phi
        self.print_steps = int(np.ceil(5e9/n)) # Prints every 5 billion steps for a single particle...

        # The variables that are used for the MPI scheduling...
        self._gpu = False
        try:
            device = xp.cuda.runtime.getDevice() # If this fails we are on the CPU
            self._gpu = True
            device = xp.cuda.Device(device)
            (free_mem, tot_mem) = device.mem_info
            # Use less than 80% of the GPU memory for all ranks combined.
            use_mem = np.floor(0.4*free_mem) # Use the 40%, as there might be 2 workers per task... 0.4*2=0.8
            use_mem_single = use_mem/mpi4py.MPI.COMM_WORLD.Get_size()
            self._chunk = use_mem_single/8 # Chunks of for r (Divide by 8, since we're using doubles...)
        except:
            # We're on CPU
            self._chunk = 8*1024/8 # Chunks of M=32Kb for r (Divide by 8, since we're using doubles...)

        self._scheduler = 0
        self._rank = 0

        self._r = None
        self._r_loc = None # r_loc might be unused, depending on the scheduling technique selected...

        self._f = None
        self._f_loc = None

        """ The client should make sure that the estimate method has terminated at least once before calling the variables directly
        """
        self.mean = None
        self.variance = None

        # By default uses 1 worker per rank...
        self._strategy = self._call_simple

    # Id is an MPI tag to receive the average synchronously. The rest is gathered asynchnously from the communicator
    def __call__(self, n_loc: int, integrator, gen ):
        return self._strategy(n_loc, integrator, gen)

    def _call_simple(self, n_loc: int, integrator, gen):
        # Create a random number generator to decide randomly when to print...
        print_gen = xp.random.default_rng()

        # Start the real computations
        cur = xp.empty( (n_loc, self.dim) )
        cur[:,:] = xp.copy(self.x0)[None,:]

        ii = 1
        while ((ii-1)*self.print_steps) < self.steps:
            start = (ii-1)*self.print_steps+1
            end = min(ii*self.print_steps, self.steps)
            for step in range(start,end+1):
                cur = integrator.step(cur, gen)
            ii += 1
            # Try to avoid printing too often...
            if print_gen.random() < 1/(self.n*self.steps):
                print(f"Step {step}/{self.steps}")

        cur_f = self.phi(cur)

        self._r_loc = cur
        self._f_loc = cur_f

        res = ( xp.sum(cur_f)/self.n, xp.sum(cur_f*cur_f)/self.n )
        return res

    def _call_complex(self, n_loc: int, integrator, gen):
        # Create a random number generator to decide randomly when to print...
        print_gen = xp.random.default_rng()

        # Start the real computations
        cur = xp.empty( (n_loc, self.dim) )
        cur[:,:] = xp.copy(self.x0)[None,:]

        ii = 1
        while ((ii-1)*self.print_steps) < self.steps:
            start = (ii-1)*self.print_steps+1
            end = min(ii*self.print_steps, self.steps)
            for step in range(start,end+1):
                cur = integrator.step(cur, gen)
            ii += 1
            # Try to avoid printing too often...
            if print_gen.random() < 1/(self.n*self.steps):
                print(f"Step {step}/{self.steps}")

        cur_f = self.phi(cur)

        res = ( xp.sum(cur_f)/self.n, xp.sum(cur_f*cur_f)/self.n, cur, cur_f )
        return res

    def estimate(self, steps) -> Tuple[float, float]:
        self.steps = steps
        self.dim = len(self.x0) # x0 is suposed to be a vector, get the dimensions from that...

        # In the case of GPU, we don't care about the real workload, but only the memory usage
        workload = self.dim*self.n

        # print('Integrating the results...')
        # Get the communicator before scheduling...
        comm = mpi4py.MPI.COMM_WORLD
        # Our root will be rank scheduler...
        rank = comm.Get_rank()
        size = comm.Get_size()
        self._rank = rank

        # The squares of result (this will be broadcasted: from the root, to the other processes)
        squares_all = None # The appropriate value is None for the way in which reduce works
        # All workers need to know these two variables, even if they are not used for computations
        mean = 0.0
        squares = 0.0 # The point is to ensure that they are seen doubles.
        # How many tasks to compute?
        if workload <= self._chunk:
            # 1. Instantiate also the receiving buffers. They will be used asynchronously
            self._r_loc = xp.empty( (self.n,self.dim) )
            self._f_loc = xp.empty( (self.n) )

            if(rank == self._scheduler):
                # Too little particles, create just one task...
                print('Using just one task')
                integrator = self.i_f.getIntegrator(self.dt)
                # Store the random number generator state.
                # Note that `seed=None` uses a run-dependent start state.
                gen = xp.random.default_rng(seed=None)

                # 1. Compute MC with just one task...
                mean, squares = self(self.n, integrator, gen)

                rr = np.concatenate( (np.arange(0,self._scheduler, dtype='int'), np.arange(self._scheduler+1,size, dtype='int') ) )
                for i in rr:
                    comm.send( np.array([ True ], dtype='bool') , dest=i)
            else:
                # MPI uses busy-waiting, global warming get away.
                b = np.array([ False ], dtype='bool')
                req = comm.irecv( source=self._scheduler)
                while not b:
                    time.sleep(1)
                    success, b = req.test()

            # 2. Move r_loc and f_loc in their definitive position... This must happen for all MPI ranks...
            self._transfer_started = threading.Semaphore()
            if self._gpu:
                self._transfer_started.acquire() # On the GPU, acquire and wait for the transfer...
                t = threading.Thread(target=self._bcast_thread, args=(comm,), daemon=True)
                t.start()
            else:
                self._r = self._r_loc
                self._f = self._f_loc

                # While the result of the experiment is broadcasted, as well, they are broadcasted asynchronously...
                self.req_r = comm.Ibcast(self._r)
                self.req_f = comm.Ibcast(self._f)
        else:
            N_each = int(xp.floor(self.n/size))
            # The scheduler takes the slack on itself
            if rank == self._scheduler: # take the slack on itself...
                N_each = self.n - N_each*(size-1)

            # Instantiate the empty arrays in memory (for every rank...)
            self._r_loc = xp.empty( (N_each,self.dim) )
            self._f_loc = xp.empty( (N_each) )

            if workload <= size*self._chunk:
                # Too little particles, create just one task per rank...
                if rank==self._scheduler:
                    print(f'Using just one task per rank, {size} ranks')

                integrator = self.i_f.getIntegrator(self.dt)
                gen = xp.random.default_rng(seed=None)

                mean, squares = self(N_each, integrator, gen)
            else:
                # Too many particles are to compute: schedules tasks of N0 particles or more...
                N0 = np.floor(np.log2( self._chunk/self.dim ))
                N0 = int(2**N0)

                # Use the complex technique to return the results: the workers cannot simply write on a shared memory...
                self._strategy = self._call_complex
                tasks = int(np.ceil(N_each/N0))

                pars_all = []
                for i in range(tasks-1):
                    pars = ( N0, self.i_f.getIntegrator(self.dt), xp.random.default_rng(seed=None) )
                    pars_all.append(pars)

                pars = ( N_each - N0*(tasks-1), self.i_f.getIntegrator(self.dt), xp.random.default_rng(seed=None) )
                pars_all.append(pars)

                # Launches the tasks...
                futures = ParallelExecutor.SubmitTasks(comm, self, pars_all)

                mean = 0
                squares = 0
                start = 0
                for f in mpi4py.futures.as_completed(futures):
                    m, s, cur, cur_f = f.result()
                    mean += m
                    squares += s
                    l = len(cur[:,0])
                    end = start+l
                    ind = range(start, end)
                    start = end
                    self._r_loc[ind,:] = cur
                    self._f_loc[ind] = xp.squeeze(cur_f)

            N_each_tot = np.empty( (size),dtype=np.int64)
            comm.Allgather((np.array(N_each),mpi4py.MPI.INTEGER4),(N_each_tot,mpi4py.MPI.INTEGER4))

            self._transfer_started = threading.Semaphore()
            if self._gpu:
                self._transfer_started.acquire() # On the GPU, acquire and wait for the transfer...
                t = threading.Thread(target=self._gather_thread, args=(comm,N_each_tot))
                t.start()
            else:
                pass
                # self._r = np.empty( (self.n,2) )
                # self._f = np.empty( (self.n) )
                # self.req_r = comm.Iallgatherv(self._r_loc, (self._r,N_each_tot*self.dim))
                # self.req_f = comm.Iallgatherv(self._f_loc, (self._f,N_each_tot))

        # This is the result and as such these are blocking operations...
        if self._gpu:
            mean = xp.asnumpy(mean)
            squares = xp.asnumpy(squares)

        # Now, distribute the result to everyone... For simplicity, this is a synchronous operations, they are just numbers.
        squares_all = np.array(0,'d')
        self.mean = np.array(0,'d')
        comm.Allreduce(np.array(mean), self.mean, op=mpi4py.MPI.SUM)
        comm.Allreduce(np.array(squares), squares_all, op=mpi4py.MPI.SUM)

        self.variance = squares_all - self.mean*self.mean

        # before returning the result, start a thread that acquires the requests.
        # DO NOT Start daemon threads, in order not to cause segmentation faults.
        # self._message_received = threading.Semaphore(value=0)
        # t = threading.Thread(target=self._consume_reqs, args=(comm,))
        # t.start()

        # With MPI we get np arrays, convert to double to return...
        self.mean = float(self.mean)
        self.variance = float(self.variance)

        return self.mean, self.variance

    def get_r(self):
        with self._message_received:
            pass

        return self._r

    def get_f(self):
        with self._message_received:
            pass
        return self._f

    def _bcast_thread(self, comm:mpi4py.MPI.Comm):
        # While the result of the experiment is broadcasted, as well, they are broadcasted asynchronously...
        r_stream = xp.cuda.Stream(non_blocking=True) # Asynchronous stream
        f_stream = xp.cuda.Stream(non_blocking=True) # Asynchronous stream

        self._r = xp.asnumpy(self._r_loc, stream=r_stream, blocking=False)
        self._f = xp.asnumpy(self._f_loc, stream=f_stream, blocking=False)

        r_stream.synchronize()
        del r_stream
        self.req_r = comm.Ibcast(self._r)

        f_stream.synchronize()
        del f_stream
        self.req_f = comm.Ibcast(self._f)

        self._transfer_started.release()

        self._r_loc = None # Free the memory
        self._f_loc = None # Free the memory

        xp.get_default_memory_pool().free_all_blocks()

    def _gather_thread(self, comm:mpi4py.MPI.Comm, N_each_tot:int):
        r_stream = xp.cuda.Stream(non_blocking=True) # Asynchronous stream
        f_stream = xp.cuda.Stream(non_blocking=True) # Asynchronous stream

        self._r_loc = xp.asnumpy(self._r_loc, stream=r_stream, blocking=False)
        self._f_loc = xp.asnumpy(self._f_loc, stream=f_stream, blocking=False)

        self._r = np.empty( (self.n,2) )
        self._f = np.empty( (self.n) )

        r_stream.synchronize()
        del r_stream
        self.req_r = comm.Iallgatherv(self._r_loc, (self._r,N_each_tot*self.dim))

        f_stream.synchronize()
        del r_stream
        self.req_f = comm.Iallgatherv(self._f_loc, (self._f,N_each_tot))

        self._transfer_started.release()

        xp.get_default_memory_pool().free_all_blocks()

    def _consume_reqs(self, comm:mpi4py.MPI.Comm):
        """This method makes sure that the requests are always consumed, otherwise dealocks are possible..."""
        with self._transfer_started:
            try:
                # b = self.req_f.Test()
                # while not b:
                #     time.sleep(1)
                #     b = self.req_f.Test()
                #     print(b)

                # b = self.req_r.Test()
                # while not b:
                #     time.sleep(1)
                #     b = self.req_r.Test()
                #     print(b)

                self.req_f.wait()
                self.req_r.wait()
                self._message_received.release()
            except:
                print('WARNING: Wait on a transfer request failed.')
