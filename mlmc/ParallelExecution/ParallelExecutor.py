from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor

from mpi4py import MPI
import json

class ParallelExecutor:
    """ Allows the client to change parallelization techniques by manipulations of:
        workers:    number of workers in the pool. 
            Tries to load the configuration file ParallelExecution.conf, if none is found, uses workers=2.
        strategy:   define the Executor class to be used. default is ThreadPoolExecutor
    """

    try:
        with open('ParallelExecution.conf', "r") as f:
            opt = json.load(f)
            workers = opt['Threads_per_rank']
    except:
        workers = 2
    strategy = ThreadPoolExecutor

    _executor = None
    @contextmanager
    def SelectExecutor(comm: MPI.Comm, proc=workers):
        """ The scheduler is the only rank that will be able to launch tasks...
        """
        if ParallelExecutor._executor is None:
            with ParallelExecutor.strategy(max_workers=proc) as executor:
                yield executor

    @staticmethod
    def SubmitTasks(comm: MPI.Comm, operation, pars ):
        """ Given a target operation submits the task, returns a synchronous result.
        Operation should be a callable object of Class ParallelOperation: returns two communicators
        pars are the parameters to submit to the function: pars is a list of lists of parameters.
        To submit a single task:
        [ [par_0, ..., par_n] ]
        The function returns futures. The submittor of the tasks MUST know how to get the results from the future.
        """
        l = len(pars)
        if l < ParallelExecutor.workers:
            with ParallelExecutor.SelectExecutor(comm, proc=l) as executor:
                tasks = [ executor.submit(operation, *pars_cur) for pars_cur in pars ]
                return tasks

        with ParallelExecutor.SelectExecutor(comm) as executor:
            tasks = [ executor.submit(operation, *pars_cur) for pars_cur in pars ]
            return tasks
