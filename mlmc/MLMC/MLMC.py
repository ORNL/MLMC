import numpy as np
from mlmc.xp import xp

from typing import Tuple

from .MLMC_L import MLMC_l_AbstractFactory

import mpi4py.MPI
import mpi4py.futures

from mlmc.ParallelExecution.ParallelExecutor import ParallelExecutor

import time

# A class for MC simulations of the process
class MLMC:
    def __init__(self, MLMC_l_factory : MLMC_l_AbstractFactory) -> None:
        self.MLMC_l_fac = MLMC_l_factory

        # CAREFUL _N_task will be updated in the future depending on the load... (and the device)
        self._gpu = False
        try:
            device = xp.cuda.runtime.getDevice() # If this fails we are on the CPU
            self._gpu = True
            device = xp.cuda.Device(device)
            (free_mem, tot_mem) = device.mem_info
            # Use up to 80% of device memory for all chunks combined
            use_mem = np.floor(0.4*tot_mem) # Use the 40% in the computations, as there might be 2 workers per tasks...
            use_mem_single = use_mem/mpi4py.MPI.COMM_WORLD.Get_size()
            self._N_task = use_mem_single/8 # Chunks of for r (Divide by 8, since we're using doubles...)
            self._N_each_min = 2048
        except:
            # We're on CPU
            self._N_task = 1*1024*1024/8 # Chunks of M=1Mb=1024*1024/8 particles for P[0] (Divide by 8, since we're using doubles...)
            self._N_each_min = 16

        self._scheduler = 0
        self._rank = 0
        self._size = 0
        self._slack = []

        self._TASK_TAG = 12
        self._PARS_TAG = 45

        # The samples of MLMC at every level
        self.P = None

        self._L = 2 # starts with at least 3 levels: 0,1,2

        # Nl, cost, ml, Vl, costl_each...
        self.Nl = None
        self.ml = None
        self.ml_par = None
        self.Vl = None
        self.Vl_par = None
        self.costl = None
        self.costl_par = None
        self.costl_each = np.array([ 1, 2, 4 ]) # Just an initial guess, it will be irrelevant after the first iteration

        # Initialize the parameters...
        self._par_est = [ True, True, True ]
        self._pars = [ 0 ]*3

    def estimate(self, tol:float, N0:int=1000, L_max:int = 10, alpha:float=0, beta:float=0, gamma:float=0) -> Tuple[float, float]:
        assert L_max >= 2
        assert N0 > 0
        tol = abs(tol) # If the user gives a negative value, make it positive.

        # Get the communicator before scheduling...
        comm = mpi4py.MPI.COMM_WORLD
        # Our root will be rank scheduler...
        self._rank = comm.Get_rank()
        self._size = comm.Get_size()
        self._workers = [ j for j in range(self._size) if j!=self._scheduler] # This are the workers.

        if self._rank == self._scheduler:
            print('Computing estimate...')

        #
        # This part initializes MLMC
        #

        # Initialize shared data, using the manager above
        self._init_data(N0)
        # Parameter estimation... The scheduler takes this task for itself.
        if self._rank == self._scheduler:
            self._init_pars(alpha, beta, gamma)

        #
        # This part is the real computation method
        #
        it = 0
        dNl = self.Nl # First iteration...
        while dNl.sum() > 0:
            # Submit the new tasks, retrieve the information about the levels who will be updated.
            if self._rank == self._scheduler: # Only the scheduler schedules
                new_comp_l, Tasks_queue = self._schedule_tasks(dNl)
                futures = self._submit_tasks(comm, Tasks_queue)
            else:
                new_comp_l = None
                futures = self._submit_tasks(comm, None)

            # Erase the mean and variance of the levels that will be modified.
            new_comp_l = comm.bcast(new_comp_l, root=self._scheduler) # First, everybody must know what levels will be updated.
            self._erase_partials(new_comp_l)

            # Compute the partial mean and variance (previous experiments... at levels new_comp_l)
            self._compute_averages_ready(new_comp_l)

            # Collect the new results
            self._collect_results(comm, futures)

            # Reduce the partial information so that every rank is updated (on a need-to-know basis).
            self._reduce_quantities(comm)

            # Compute and reduce variance
            self._compute_variance(comm, new_comp_l)

            # Fix to cope with possible zero values for ml and Vl. (can happen in some applications when there are few samples)
            # Do BEFORE recomputing the regression...
            for l in np.arange(3,self._L+1):
                self.ml[l] = max( self.ml[l], 0.5*self.ml[l-1]/(2**alpha) ) # Uses alpha
                if self._rank == self._scheduler: # only the scheduler knows the variance.
                    self.Vl[l] = max( self.Vl[l], 0.5*self.Vl[l-1]/(2**beta) ) # Uses beta

            if self._rank == self._scheduler: # The scheduler orders the following things.
                # 1. Use linear regression to estimate the new alpha, beta, gamma (if not given, see function compute_pars)
                # Do this asynchronously. (Tell the thread that the parameters are not ready anymore...)
                if ( self._par_est[0] | self._par_est[1] | self._par_est[2]):
                    # This is the correct way to submit a function with no arguments.
                    pars_ready = ParallelExecutor.SubmitTasks(comm, self._compute_pars, ((),) )

                # 2. Set optimal number of additional samples (from page 273 of the 2015 survey.)
                self.costl_each = self.costl/self.Nl
                Nl_new = MLMC._get_Nl(tol, self.Vl, self.costl_each) # np.ceil( 2*np.sqrt(self.Vl/self.costl_each)*np.sum(np.sqrt(self.Vl*self.costl_each)) / (tol*tol) )

                # Ensures that we don't run into errors in the case in which the optimal number of samples dicreases
                dNl = np.maximum(Nl_new - self.Nl, 0).astype(int)

            # if (almost) converged, estimate remaining error and decide whether a new level is required
            new_level = False
            if self._rank == self._scheduler:
                alpha, beta, gamma = pars_ready[0].result()
                new_level = self._decide_next_iteration(dNl, N0, L_max, tol, alpha, beta, gamma)

            new_level = comm.bcast(new_level,root=self._scheduler)
            beta = comm.bcast(beta,root=self._scheduler)
            gamma = comm.bcast(gamma,root=self._scheduler)

            if(new_level): # everyone must rescale their quantities
                Nl_new = self._add_level(tol, beta, gamma)
            if self._rank == self._scheduler: # only the scheduler knows all the information to find dNl correctly
                dNl = self._prepare_iteration(N0, Nl_new) # This function works both wether we have a new level or not

            self.Nl = comm.bcast(self.Nl, root=self._scheduler) # All the ranks need to know the correct information to compute the correct average
            dNl = comm.bcast(dNl,root=self._scheduler) # Used to decide whether to continue or not.
            if self._rank == self._scheduler:
                print(self.ml)
                print(self.Vl)

            it += 1

        # This is to distribute the result to every rank...
        self.Vl = comm.bcast(self.Vl,root=self._scheduler)
        if self._rank == self._scheduler:
            print(self.Nl)
        
        # At this point the user might even call get_pars()
        self._pars = comm.bcast(self._pars, root=self._scheduler) 
        
        # Consider whether to move the experiments and all the other quantities to synchronize all ranks completely.
        return sum(self.ml), sum(self.Vl/self.Nl)

    def get_pars(self, ):
        """ To be called only after estimate has ended
        """
        return np.copy(np.array(self._pars))

    def _init_data(self, N0:int) -> None:
        L0 = self._L+1 # initial number of levels (0,1,2), 2+1=3 levels.
        # Delicate: initialize a shared list... of shared lists.
        self.P = [ [] ]*L0 # Parent list...
        for l in np.arange(L0):
            self.P[l] = [] # Child list...

        # Initialize the structures needed for the estimations
        self.costl_par = np.zeros( (L0) )
        self.Vl_par = np.zeros( (L0) )
        self.ml_par = np.zeros( (L0) )
        self.ml = np.zeros( (L0) ) # Every rank needs the mean to compute the variance then.

        # The scheduler must know about the full cost, as well
        if self._rank == self._scheduler:
            self.costl = np.zeros( (L0) )
            self.Vl = np.zeros( (L0) )

        self.Nl = N0*np.ones( (L0), int)

    def _init_pars(self, alpha:float, beta:float, gamma:float):
        if alpha > 0:
            self._par_est[0] = False
            self._pars[0] = alpha
        if beta > 0:
            self._par_est[1] = False
            self._pars[1] = beta
        if gamma > 0:
            self._par_est[2] = False
            self._pars[2] = gamma
            self.costl_each = 2**(gamma*np.arange(3))

    def _schedule_tasks(self, dNl:np.ndarray):
        L = len(dNl) # levels are 0,1,...,L-1
        # Load balancing using information about the cost per level, normalized with the cost of the last level.
        tot = np.ceil(dNl*self.costl_each/self.costl_each[L-1]) # total amount of computations for current iteration

        # This for loop decides what levels need more computations.
        new_comp_l = []
        for l in range(L): # iterates from 0 to L-1.
            if tot[l] > 0:
                new_comp_l.append(l)

        # This for loops iterate only over the level who require more computations
        # According to the information about costl_each, creates fixed size tasks
        Tasks_queue = [] # This is the global tasks queue. It will be scattered around among the ranks
        for l in new_comp_l:
            tsk = int( np.ceil(tot[l]/self._N_task) ) # Number of fixed size tasks for this level.
            tsk_min = int( np.floor(tot[l]/self._N_each_min) )
            tsk = max(tsk, tsk_min)
            N_each = int( np.floor(dNl[l]/tsk) )
            last = int( dNl[l] - N_each*(tsk-1) )

            for i in range(tsk-1):
                pars = ( l, N_each )
                Tasks_queue.append(pars)
            Tasks_queue.append( (l,last) )
        return new_comp_l, Tasks_queue

    def _submit_tasks(self, comm:mpi4py.MPI.Comm, Tasks_queue:list):
        if self._size == 1: # If there is a single rank, that rank takes on the whole queue
            Tasks_queue_loc = Tasks_queue
        elif self._rank == self._scheduler:
            l = len(Tasks_queue)
            l_loc = int(np.floor(l/self._size)) # Every rank will receive this amount of tasks. Slacks on the last ranks

            # Some ranks receives more...
            remainder = l-l_loc*self._size
            sr = self._size-remainder
            ind = np.random.permutation(l).astype(int) # Shuffle the indices so that likely everybody will take more or less the same cost and all the levels.

            self._slack = self._workers[sr-1:] # slacks on this workers.
            cur = ind[0:l_loc] # This is for ii=0
            Tasks_queue_loc = [ Tasks_queue[j] for j in cur ] # Takes this for itself...
            reqs = [ None ]*(sr-1)
            for i in range(sr-1):
                cur = ind[slice((i+1)*l_loc,(i+2)*l_loc)]
                reqs[i-1] = comm.isend( [ Tasks_queue[j] for j in cur ] , dest=self._workers[i], tag=self._TASK_TAG)

            cur_slack = sr*l_loc
            for i in range(len(self._slack)):
                cur = ind[slice( cur_slack+i*(l_loc+1), cur_slack+(i+1)*(l_loc+1) )]
                comm.send( [ Tasks_queue[j] for j in cur ] , dest=self._slack[i], tag=self._TASK_TAG) # Send the last ones synchronously...

            # When using isend it is not safe to leave the requests unchecked, apparently
            mpi4py.MPI.Request.waitall(reqs)
        else:
            Tasks_queue_loc = comm.recv(source=self._scheduler, tag=self._TASK_TAG)

        if self._size > 1:
            self._slack = comm.bcast(self._slack, root=self._scheduler) # Everyone should know this variable.

        if len(Tasks_queue_loc):
            # Returns the futures onto which every rank will synchronize.
            return ParallelExecutor.SubmitTasks(comm,self._compute_experiments,Tasks_queue_loc)

        # No task to launch, nothing to return
        return ()

    def _compute_experiments(self, l:int, N:int) -> Tuple[ int, np.ndarray, int ]:
        mlmc_l = self.MLMC_l_fac.create(l)
        P_l_par, cost_l_par = mlmc_l.compute(N) # Partial solution for level l
        return l, P_l_par, cost_l_par

    def _erase_partials(self, changed:list):
        # Delete completely the prior information.
        self.ml[:] = 0 # keep the same array, just put everything to zero. Do not reinstantiate.

        #  The partials are not touched if the level doens't change. In this way they will reduce to the current result again.
        self.ml_par[changed] = 0
        self.Vl_par[changed] = 0

    def _compute_averages_ready(self, changed:list):
        # This cycle computes the mean for the experiments already available...
        for l in changed:
            for P_cur in self.P[l]:
                self.ml_par[l] += np.sum(P_cur/self.Nl[l])

    def _collect_results(self, comm:mpi4py.MPI.Comm, futures: list):
        """ futures is the list of futures for the submitted tasks
        """
        for f in mpi4py.futures.as_completed(futures):
            l, P_par, cost = f.result()
            if self._gpu:
                s = xp.cuda.Stream(non_blocking=True) # Asynchronous stream
                P_par = xp.asnumpy(P_par,stream=s,blocking=False)

            self.costl_par[l] += cost

            if self._gpu: # Even thought the call was asynchronnous, get it ogg the GPU asap
                 s.synchronize()
                 del s # free the memory
                 xp.get_default_memory_pool().free_all_blocks()


            # Free the memory
            del f

            self.ml_par[l] += np.sum(P_par/self.Nl[l]) # Do this on the CPU for simplicity
            self.P[l].append(P_par)

        # We slack on the last tasks... At this point assume that the last rank can signal to the sleepy ones that they can now wake up.
        if self._size > 1 and len(self._slack) > 0:
            ls = len(self._slack)
            if self._rank in self._slack:
                comm.send( np.array([ True ], dtype='bool') , dest=self._scheduler)
            elif self._rank == self._scheduler:
                b = np.array([ False ]*ls, dtype='bool')
                req = [ None ]*ls
                for i in range(ls):
                    req[i] = comm.irecv( source=self._slack[i] )
                while not np.prod(b):
                    time.sleep(1)
                    for i in range(ls):
                        success, bi = req[i].test()
                        if bi:
                            b[i] = True

                remaining = [ i for i in range(self._size) if (i!=self._scheduler and i not in self._slack) ]
                for i in remaining:
                    comm.send( np.array([ True ], dtype='bool') , dest=i)
            else:
                # MPI uses busy-waiting, global warming get away.
                b = np.array([ False ], dtype='bool')
                req = comm.irecv( source=self._scheduler)
                while not b:
                    time.sleep(1)
                    success, b = req.test()

    def _reduce_quantities(self, comm:mpi4py.MPI.Comm):
        # The changed variables are updated, whereas the other will reduce to the previous value again.
        comm.Allreduce(self.ml_par,self.ml, op=mpi4py.MPI.SUM) # This must be reduced and make known to every one. Used for variance.
        cost_new = np.zeros((self._L+1))
        comm.Reduce(self.costl_par,cost_new, op=mpi4py.MPI.SUM, root=self._scheduler)
        if self._rank == self._scheduler:
            self.costl = cost_new

    def _compute_variance(self, comm:mpi4py.MPI.Comm, changed:list):
        for l in changed:
            for P_cur in self.P[l]:
                diff = P_cur - self.ml[l]
                self.Vl_par[l] += np.sum( (diff*diff)/self.Nl[l] )

        if(self._rank == self._scheduler):
            self.Vl[:] = 0 # Do not reinstantiate, just reassing values...

        comm.Reduce(self.Vl_par,self.Vl, op=mpi4py.MPI.SUM, root=self._scheduler)

    @staticmethod
    def _get_Nl(tol: float, Vl:np.ndarray, costl_each:np.ndarray):
        Nl_new = np.ceil( 2*np.sqrt(Vl/costl_each)*np.sum(np.sqrt(Vl*costl_each)) / (tol*tol) )
        return Nl_new
    
    def _decide_next_iteration(self, dNl:np.ndarray, N0:int, L_max:int, tol:float, alpha:float, beta:float, gamma:float) -> bool:
        """ Returns whether it is necessary/convenient to add a new level.
        """
        exp_cost = sum(dNl*self.costl_each)
        # if (sum(dNl*self.costl_each/self.costl_each[0]) < self._N_task*self._size):
        if (exp_cost > N0*self.costl_each[-1]*2**gamma):
            range = np.arange(3)
            range = self._L-range
            ml_loc = self.ml # like before
            ml_loc = ml_loc[range] # wrap the relevant part of ml...

            rem = max( abs(ml_loc) / (2**(range*alpha)) ) / (2**alpha - 1)

            # If the estimated bias is big, increase the level.
            if rem > tol/np.sqrt(2):
                if (self._L==L_max):
                    if (sum(dNl*self.costl_each/self.costl_each[0]) < N0):
                        e = RuntimeError('failed to achieve weak convergence in ' + str(L_max) + ' levels')
                        raise e
                    return False
                return True
            elif(self._L<L_max):
                # If adding a level costs (in expectation) as much or less than continuing with the current levels...
                Vl = np.append( self.Vl, self.Vl[-1]/(2**beta) )
                costl_each = np.append( self.costl_each, self.costl_each[-1]*2**gamma )
                Nl_new = MLMC._get_Nl(tol, Vl, costl_each)

                dNl_new = np.zeros( (self._L+2) ) # 0, 1, ... _L, _L+1. We count from 0, so it is L+2 positions...
                dNl_new[0:-1] = Nl_new[0:-1] - self.Nl
                dNl_new[-1] = max(Nl_new[-1], N0)
                dNl_new = np.maximum(dNl_new, 0).astype(int)

                exp_cost_new = sum(dNl_new*costl_each)
                # Assume that in general is very convenient to add levels. This allows for more parallelization and less iterations.
                if exp_cost_new <= 10*exp_cost: 
                    return True
            return False
        return False

    def _add_level(self, tol:float, beta:float, gamma:float):
        self._L = self._L+1
        # Delicate: initialize a shared list... of shared lists.
        self.P.append([]) # Parent list...

        # Initialize the structures needed for the estimations
        self.costl_par = np.append( self.costl_par, 0 )
        self.Vl_par = np.append( self.Vl_par, 0 )
        self.ml_par = np.append( self.ml_par, 0 )
        self.ml = np.append( self.ml, 0 ) # Every rank needs the mean to compute the variance then.

        # The scheduler must know about the full cost, as well
        if self._rank == self._scheduler:
            self.costl = np.append( self.costl, 0 )
            self.Vl = np.append( self.Vl, self.Vl[-1]/(2**beta) )
            self.costl_each = np.append( self.costl_each, self.costl_each[-1]*2**gamma ) # This is an expectation
            self.Nl = np.append( self.Nl, 0 )
            # Recompute Nl_new taking into account the new level.
            Nl_new = MLMC._get_Nl(tol, self.Vl, self.costl_each)
            
            return Nl_new
        return 0 # Not the scheduling task, clearly.

    def _prepare_iteration(self, N0:int, Nl_new:np.ndarray):
        # This is to make sure that a possible new level computes at least N0 elements.
        Nl_new[-1] = max(Nl_new[-1], N0)
        dNl = Nl_new - self.Nl

        self.Nl = np.maximum(self.Nl, Nl_new).astype(int)
        dNl = np.maximum(dNl, 0).astype(int)

        return dNl

    def _compute_pars(self) -> Tuple[float, float, float]:
        L = self._L
        A = MLMC._get_A(L)

        if self._par_est[0]:
            self._compute_alpha(A)
        if self._par_est[1]:
            self._compute_beta(A)
        if self._par_est[2]:
            self._compute_gamma(A)

        return self._pars # They are either changed or unchanged based on self.pars_est

    def _get_A(L:int):
        A = np.ones( (L,2) )
        A[:,0] = np.arange(1,L+1)
        return A

    def _compute_alpha(self, A):
        b = np.log2( np.abs(self.ml[1:]))
        x = np.linalg.lstsq(A,b, rcond=None)

        self._pars[0] = max( 0.5, -x[0][0] )

    def _compute_beta(self, A):
        b = np.log2( self.Vl[1:] )
        x = np.linalg.lstsq(A,b, rcond=None)
        self._pars[1] = max( 0.5, -x[0][0] )

    def _compute_gamma(self, A):
        b = np.log2( self.costl_each[1:] )
        x = np.linalg.lstsq(A,b, rcond=None)
        self._pars[2] = max( 0.5, x[0][0] )
