"""
Provides a QueueSimulator class which allows for a simulation of a Poisson process with a discrete number of queues (n)
to handle each arrival. For example, patients arriving at a medical clinic with n doctors.
"""

__all__ = ['QueueSimulator']
__version__ = ['0.0.1']
__author__ = ['Jason Preszler <jpreszler@gmail.com>']

import numpy as np
import pandas as pd
import scipy.stats as stats
import datetime

class QueueSimulator:
    def __init__(self, numQueues, rate, start_hour, end_hour, appt_low, appt_high):
        """ Creates QueueSimulator object.

        Arguments
        ------
        numQueues: integer number of queues to handle arrivals
        rate: float, 'scale' parameter of exponential distribution, time between arrivals
        start_hour: integer 0:24, hour of day that arrivals can start
        end_hour: integer 0:24, hour of day that arrivals stop (no arrivals after, last arrival may be before)
        appt_low: integer, lower bound of number of minutes needed to process an arrival
        appt_high: integer, upper bound of number of minutes needed to process an arrival

        Description
        ------
        This class will simulate a Poisson process where items arrive between 'start_hour' and 'end_hour',
        with an item arriving every 'rate' minutes on average. Each arrival requires between 'appt_low'
        and 'appt_high' minutes to be processed (uniformly random) and there are 'numQueues' available
        to process the arrivals.

        Example
        -----
        'simulation = QueueSimulator(3, 10, 9, 16, 5, 21)'
        can be thought of as a clinic with 3 doctors, patients arriving every 10 minutes (exponentially dist.)
        between 9AM and 4PM, and each patient requires an appointment between 5 and 20 minutes (uniformly dist.).
        The appointment length is a property of each patient, and varies accordingly.

        Calling 'simulation.run_simulation(1)' will simulate one day at the clinic. Results stored in the
        dataframe 'simulation.results'
        """

        self.rate = rate
        self.numQueues = numQueues
        self.start = datetime.datetime.combine(datetime.date.today(), datetime.time(start_hour,0,0))
        self.end = datetime.datetime.combine(datetime.date.today(), datetime.time(end_hour,0,0))
        self.appt_low = appt_low
        self.appt_high = appt_high
        minutes_for_new_items = (end_hour-start_hour)*60 #new patients seen between 9AM and 4PM
        time_between_items = rate #exponential dist. time parameter
        self.expected_count = int(np.ceil(stats.poisson.ppf(.9999, minutes_for_new_items/time_between_items)))
        self.ques = [datetime.datetime.combine(datetime.datetime.today(), datetime.time(start_hour,0,0)) for i in range(0, self.numQueues)]
        cols = ['simulation', 'num_items', 'wait_count', 'avg_wait_time', 'close_time']
        self.results = pd.DataFrame(columns = cols)
        return

    def __single_sim_results(self):
        itemID = np.arange(0, self.expected_count)
        minutes_to_arrival = np.random.exponential(scale = self.rate, size = self.expected_count)
        arrival_times = [self.start+datetime.timedelta(minutes = i) for i in minutes_to_arrival.cumsum()]
        arrivals = pd.DataFrame({'id':itemID, 'min_btwn_arrival': minutes_to_arrival, 'arrival_time': arrival_times, 'appt_length': np.random.uniform(low=self.appt_low, high=self.appt_high, size=self.expected_count)})
        arrivals = arrivals[arrivals['arrival_time']<=self.end]
        arrivals['appt_length_minutes'] = arrivals.appt_length.apply(lambda x: datetime.timedelta(minutes=x))
        arrivals['queue'] = np.nan
        arrivals['wait_time'] = datetime.timedelta(minutes=0)
        arrivals['appt_start_time'] = arrivals['arrival_time']
        arrivals['appt_end_time'] = arrivals['arrival_time']+arrivals['appt_length_minutes']
        arrivals = arrivals.apply(lambda x: self.__wait_time_update(x), axis=1)
        return(arrivals)

    def __wait_time_update(self, item):
        first_que_avail = self.ques.index(min(self.ques))
        item['queue'] = first_que_avail
        if(self.ques[first_que_avail]<= item['arrival_time']):
            self.ques[first_que_avail] = item['appt_end_time']
        else:
            #patient will wait
            item['wait_time'] = self.ques[first_que_avail] - item['arrival_time']
            item['appt_end_time'] = item['appt_end_time']+item['wait_time']
            self.ques[first_que_avail] = item['appt_end_time']
        return(item)

    def run_simulation(self, number_runs = 1):
        """Runs the Simulation

        Arguments
        -----
        number_runs: positive integer, (default 1) number of 'days' to simulate.

        Results
        -----
        Stores resulting information in 'results' dataframe attribute containing:
            num_items: the number of arrivals processed
            wait_count: the number of arrivals that had to wait before processing
            avg_wait_time: the mean time arrivals had to wait
            close_time: the time the last item was finished processing, can be before 'end_hour'
                if no arrivals can near the end.
        Each row of this dataframe is for one run of the simulation ('number_runs' is the number
        of rows).
        """
        for i in range(0, number_runs):
            self.ques = [self.start for i in range(0, self.numQueues)]
            run = self.__single_sim_results()
            run_results = pd.DataFrame({'simulation':i,
                'num_items': len(run),
                'wait_count': len(run[run['wait_time']>datetime.timedelta(seconds=0)]),
                'avg_wait_time': run.wait_time.mean(),
                'close_time': max(run['appt_end_time'])}, index=[i])
            self.results = pd.concat([self.results, run_results], ignore_index=True)
            self.results['last_appt_to_close_minutes'] = (self.results['close_time']-self.end).dt.total_seconds().div(60)
        return
