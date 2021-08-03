# Import routines

import numpy as np
import math
import random
from itertools import permutations


# Defining hyperparameters
m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger
lamda_locA=2    #lambda for poisson distribution
lamda_locB=12
lamda_locC=4
lamda_locD=7
lamda_locE=8

class CabDriver():

    def __init__(self):
        """initialise your state and define your action space and state space"""
        self.action_space = [(0, 0)] + \
            list(permutations([i for i in range(m)], 2))
        self.state_space = [[a, b, c] for a in range(m) for b in range(t) for c in range(d)]
        self.state_init = random.choice(self.state_space)

        # Start the first round
        self.reset()


    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        state_encod = [0 for _ in range(m+t+d)]
        state_encod[self.state_get_loc(state)] = 1
        state_encod[m+self.state_get_time(state)] = 1
        state_encod[m+t+self.state_get_day(state)] = 1
        return state_encod


    # Use this function if you are using architecture-2 
    # def state_encod_arch2(self, state, action):
    #     """convert the (state-action) into a vector so that it can be fed to the NN. This method converts a given state-action pair into a vector format. Hint: The vector is of size m + t + d + m + m."""

    #    state_encod = np.zeros(m+t+d+m+m)
    #    state_encod[m] = 1
    #    state_encod[m+t] = 1
    #    state_encod[m+t+d] = 1    
    #    state_encod[m+t+d+m-1] = 1 
    #    state_encod[m+t+d+m-1+m-1] = 1         
    #    return state_encod


    ## Getting number of requests

    def requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        location = state[0]
        if location == 0:
            requests = np.random.poisson(lamda_locA)
        elif location == 1:
            requests = np.random.poisson(lamda_locB)
        elif location == 2:
            requests = np.random.poisson(lamda_locC)
        elif location == 3:
            requests = np.random.poisson(lamda_locD)
        else:
            requests = np.random.poisson(lamda_locE)

        if requests >15:
            requests =15

        possible_actions_index = random.sample(range(1, (m-1)*m +1), requests) + [0] # appending  index 0 for "No Ride" Option
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   



    def reward_func(self, idle_time, actual_ride_time, ride_prep_time):
        """Takes in all the time values and returns the reward"""
        # If idle time is 1, then the action is (0, 0) which implies driver has chosen "No Ride" Option
        if idle_time == 1:
            reward = -C
        else:
            # Driver has chosen an action which is not a "No Ride" option, accordingly reward is calculated
            reward = (R * actual_ride_time) - (C * (actual_ride_time + ride_prep_time))
        return reward


    def update_hour_day(self, ride_time, curr_hour, curr_day):
        """
        Takes in the ride time, current hour and day and returns the updated hour and day.
        """    
        ride_time = int(ride_time)
        if (ride_time + curr_hour) < 24:
            # day is unchanged
            new_hour_of_day = ride_time + curr_hour
            new_day_of_week = curr_day
        else:
            # converting the new hour within the range (0-23)
            new_hour_of_day = (ride_time + curr_hour) % 24
            
            # Get the number of days
            num_days = (ride_time + curr_hour) // 24
            
            # Convert the day to valid range (0-6)
            new_day_of_week = (curr_day + num_days ) % 7
            
        return new_hour_of_day, new_day_of_week

    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state along with time taken for different types of rides"""
        
        next_state = []
        
        curr_loc = self.state_get_loc(state)
        pickup_loc = self.action_get_pickup(action)
        drop_loc = self.action_get_drop(action)
        
        hour_of_day = self.state_get_time(state)
        day_of_week = self.state_get_day(state)
        
        #print("curr_loc {0}, pickup_loc {1}, drop_loc {2} hour_of_day {3} day_of_week {4}".format(curr_loc, pickup_loc, 
        #                                                                         drop_loc, hour_of_day, day_of_week))
        # All the time valuesa are initialised to 0
        idle_time = 0
        actual_ride_time = 0
        ride_prep_time = 0
        
        """
         3 Scenarios: 
           1) Choose "No Ride" Option, ie refusing all the requests
           2) Pick up is same as the current location, ie the driver is already at pick up point
           3) Pick up is not same as the current location, ie the driver is in diff location than the pick up point
        """   
        
        if ((pickup_loc == 0) and (drop_loc == 0)):
            # Driver will spend the next hour idle_time till the next requests arrive
            idle_time = 1
            dest_loc = curr_loc
        elif (curr_loc == pickup_loc):
            # Current location is same as the pick up location
            actual_ride_time = Time_matrix[pickup_loc][drop_loc][hour_of_day][day_of_week]
            dest_loc = drop_loc
        else:
            # Current location is different from the pickup location
            # So ride preparation time is calculated from current loc to pick up loc first
            ride_prep_time = Time_matrix[curr_loc][pickup_loc][hour_of_day][day_of_week]
            hour_updated, day_updated = self.update_hour_day(ride_prep_time, hour_of_day, day_of_week)
            
            # Actual ride time is then calculated based on the updated hour and day
            actual_ride_time = Time_matrix[pickup_loc][drop_loc][hour_updated][day_updated]
            dest_loc = drop_loc
            
        total_time_taken = idle_time + actual_ride_time + ride_prep_time
        hour_updated, day_updated = self.update_hour_day(total_time_taken, hour_of_day, day_of_week)

        next_state = [dest_loc, hour_updated, day_updated]

        return next_state, idle_time, actual_ride_time, ride_prep_time


    def step(self, curr_state, curr_action, TimeMatrix):
        """
        Given the current state, action and the time matrix, returns the next state, reward and the time taken
        """    
        next_state, idle_time, actual_ride_time, ride_prep_time = self.next_state_func(curr_state, curr_action, TimeMatrix)
        reward = self.reward_func(idle_time, actual_ride_time, ride_prep_time)
        time_taken_for_current_ride = idle_time + actual_ride_time + ride_prep_time
        
        return next_state, reward, time_taken_for_current_ride

    def reset(self):
        """Return the current state and action space"""    
        return self.action_space, self.state_space, self.state_init

    def state_get_loc(self, state):
        return state[0]

    def state_get_time(self, state):
        return state[1]

    def state_get_day(self, state):
        return state[2]

    def action_get_pickup(self, action):
        return action[0]

    def action_get_drop(self, action):
        return action[1]
