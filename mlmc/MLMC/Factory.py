from abc import ABC, abstractmethod

class IntegratorFactory(ABC):
    """ Might be either a Factory or a Builder depending on circumstances.
        The only method here will be an instance method.
        If a "static factory" method is preferred, consider making the subclass
        a singleton.
    """

    @abstractmethod
    def getIntegrator(dt:float):
        """
        The returned object "o" must possess the method
            o.step: the next integration step
            dt is the integration step
        """
        pass
