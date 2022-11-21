import numpy as np
import sys
import copy
from arguments import args

args = args()

CAP = int(args.CAP)
INIT_T = float(args.init_T)
FINAL_T = float(args.final_T)
RAND_INIT_N_STEPS_SA = int(args.RAND_INIT_N_STEPS_SA)
N_STEPS = int(args.N_STEPS)
MAX_DIST = float(args.MAX_DIST)


class Job:
    def __init__(self, id, loc, x, y, weight):
        # id is just the id, loc is the location in the dist matrix (dist_time)
        self.id, self.loc, self.x, self.y, self.weight = id, loc, x, y, weight


class DistTime:
    def __init__(self, dist, time):
        self.dist, self.time = dist, time


class Vehicle:
    def __init__(
        self,
        cap,
        start_loc,
        end_loc,
        fee_per_dist,
        fixed_cost,
        hcpw,
        max_stops,
        max_dist,
    ):
        self.cap = cap  # Float; Change name to capacity
        self.start_loc = start_loc  # Integer
        self.end_loc = end_loc  # Integer
        self.fee_per_dist = fee_per_dist  # Float
        self.fixed_cost = fixed_cost  # Float
        self.handling_cost_per_weight = hcpw  # Float
        self.max_stops = max_stops  # Integer
        self.max_dist = max_dist  # Float


class State:
    def __init__(self, loc):
        self.dist = 0.0  # Float
        self.time = 0.0  # Float
        self.stops = 0  # Integer
        self.loc = loc  # Integer
        self.weight = 0.0  # Float


class Tour:
    def __init__(self, v):
        self.tour = []  # Vector of integers; with_capacity(cap)
        self.vehicle = v  # Class Vehicle()
        self.states = []  # Vector of class State(); with_capacity(cap)

    def calc_delta(self, delta_d, delta_w):
        """Calculates the distance "delta". When the vehicle.fee_per_dist==1, vehicle.fixed_cost==0,
        and the vehicle.handling_cost_per_weight==0 (standard values for the problem we are using),
        then this function returns its parameter delta_d.
        Parameters:
            delta_d: A distance.
            delta_w: job.weight (demand of a node).

        Returns:
            delta_d
        """
        delta = (
            delta_d * self.vehicle.fee_per_dist
            + delta_w * self.vehicle.handling_cost_per_weight
        )

        if len(self.tour) == 0:
            delta += self.vehicle.fixed_cost

        return delta  # Float

    def calc_cost(self):
        """Calculates the cost of a tour. With the current parameters (vehicle.fee_per_dist=1,
        vehicle.handling_cost_per_weight=0, vehicle.fixed_cost=0), this cost= tour.state[-1].dist.

        Returns: cost = last_state.dist (with current parameters).
        """
        last_state = self.states[-1]
        cost = (
            last_state.dist * self.vehicle.fee_per_dist
            + last_state.weight * self.vehicle.handling_cost_per_weight
            + self.vehicle.fixed_cost
        )
        return cost  # Float


class VehicleManager:
    def __init__(self, vehicles, dist_time):
        self.vehicles = vehicles  # Vector of class Vehicle()

    def alloc(self, job):
        """Returns the Vehicle closer to the node "job". With the current parameters there is only one vehicle."""
        return self.vehicles[0]


class Solution:
    def __init__(self, jobs, vm):
        self.vm = vm  # Vehicle Manager
        self.jobs = jobs  # Vector of class Job()
        self.tours = []  # Vector of class Tour(); with_capacity(100)
        self.absents = list(range(0, len(jobs)))
        self.cost = 0  # Costo de la solución.

    def Display(self):
        """Displays the cost, the number of tours, and the number of "absents" in the solution. }"""
        f = {
            "cost": self.cost,
            "vehicles": len(self.tours),
            "absents": len(self.absents),
        }
        print(f)


class Recreate:
    def __init__(self, dist_time, cost_per_absent):
        self.dist_time = dist_time  # Vector of vectors of class DistTime ([[DistTime]])
        self.cost_per_absent = cost_per_absent  # Float

    def calc_cost(self, solution):
        """Calculates (inplace) the cost of the solution as:
        solution.cost = sum(cost of each tour) + (cost per absent)*(# of absents)
        """
        cost = sum([x.calc_cost() for x in solution.tours])
        cost += self.cost_per_absent * len(solution.absents)
        solution.cost = cost

    def check_states(self, solution):
        tours = solution.tours
        for tour in tours:
            self.check_tour_states(solution.jobs, tour)

    def check_tour_states(self, jobs, t):
        tour = t.tour
        vehicle = t.vehicle

        weight = 0.0
        prev_loc = vehicle.start_loc

        for x in tour:
            job = jobs[x]
            dist_time = self.dist_time[prev_loc][job.loc]

            weight += job.weight

            prev_loc = job.loc

        dist_time = self.dist_time[prev_loc][vehicle.end_loc]

    def create_tour_states(self, jobs, t):
        """Creates new states in the tour."""
        tour = t.tour
        vehicle = t.vehicle

        # If there aren't any states yet, create one.
        if len(t.states) == 0:
            first_state = State(loc=vehicle.start_loc)
            t.states.append(first_state)

        # Creates the states that are missing.
        for x in tour[len(t.states) - 1 :]:
            state = t.states[-1]
            j = jobs[x]
            new_state = copy.deepcopy(state)
            dist_time = self.dist_time[state.loc][j.loc]

            new_state.dist += dist_time.dist
            new_state.time += dist_time.time

            if state.loc == j.loc:
                new_state.stops += 0
            else:
                new_state.stops += 1

            new_state.weight += j.weight
            new_state.loc = j.loc

            t.states.append(new_state)

        state = t.states[-1]
        last_state = copy.deepcopy(state)
        dist_time = self.dist_time[state.loc][vehicle.end_loc]
        last_state.dist += dist_time.dist
        last_state.time = dist_time.time
        last_state.loc = vehicle.end_loc

        t.states.append(last_state)

    def create_states(self, solution):
        """Create_tour_states() for each tour in Solution"""
        for t in solution.tours:
            # Each tour has len(tour.tour)+2 number of states (because it starts and then ends in depot),
            # if this isn't the case, then there are states that are yet to be created.
            if len(t.tour) + 2 > len(t.states):
                self.create_tour_states(solution.jobs, t)

    def try_insert_tour(self, tour, job):
        """Returns the index of the place in the tour where inserting the job would cause the smallest change
        in distance, and this change in distance. That is, inserting between states with indices j and j+1, and the index j.

        If the job cannot be inserted (its the demand exceeds the vehicle accumulated capacity), the index returned is
        j=0 and the distance returned is sys.maxsize.
        """
        states = tour.states
        v = tour.vehicle
        last_state = states[-1]
        pos = 0
        best = sys.maxsize

        # If the vehicle cannot take "job" because the demand exceeds its capacity.
        if (
            job.weight + last_state.weight > v.cap
        ):  # Is last_state.weigth the accumulated demands of the nodes visited?
            return (0, sys.maxsize)

        # This creates adjacent states to compare: w[j] and w[j+1]:
        # w = (states[0], states[1]); w = (states[1], states[2]); ...w = (states[n-1], states[n]);
        for j, w in enumerate(zip(states[:-1], states[1:])):

            # If both states' locations (.loc) are different from job.loc:
            # (If the node "job" hasn't been visited.)
            if w[0].loc != job.loc and w[1].loc != job.loc:
                delta_stops = 1
            else:
                delta_stops = 0

            # A restriction for if the vehicle has a maximum number of stops (not used).
            if v.max_stops > 0 and delta_stops + last_state.stops > v.max_stops:
                continue

            dt1 = self.dist_time[w[0].loc][job.loc]  # Distance between state_i and job
            dt2 = self.dist_time[job.loc][
                w[1].loc
            ]  # Distance between job and state_(i+1)
            dt3 = self.dist_time[w[0].loc][
                w[1].loc
            ]  # Distance between state_i and state_(i+1)

            # The extra distance a vehicle would have to travel if a node is inserted between two states in a tour.
            delta_d = dt1.dist + dt2.dist - dt3.dist

            # If there is a restriction of maximum distance that a vehicle can travel (not used).
            if v.max_dist > 0 and delta_d + last_state.dist > v.max_dist:
                continue

            delta_cost = tour.calc_delta(
                delta_d, job.weight
            )  # With the parameters used, delta_cost = delta_d

            if delta_cost < best:
                pos = j
                best = delta_cost

        return (pos, best)

    def try_insert(self, tours, job):
        """Tries to insert the node "job" in the tour and in the position where it causes the least
        increase in distance. If the job cannot be inserted in any of the tours (its the demand exceeds
        the vehicle's accumulated capacity), the position returned is (0,0) and the distance returned is
        "sys.maxsize."

        Returns:
            pos[0]: The tour where the job can be inserted.
            pos[1]: The position in this tour where the job can be inserted.
            best: The difference in distance that inserting this job causes.

        """
        pos = (0, 0)
        best = sys.maxsize

        for i, tour in enumerate(tours):
            j, delta = self.try_insert_tour(tour, job)
            if delta < best:
                best = delta
                pos = (i, j)

        return (pos[0], pos[1], best)  # Quitar paréntesis? Checar

    def do_insert(self, jobs, tour, j, x):
        """Inserts the absent node "x" at index "j" of the tour, then eliminates tour states
        leaving only the first "j" elements, and then it creates new states via create_tour_states(jobs, tour).
        """
        tour.tour.insert(j, x)
        del tour.states[j:]
        self.create_tour_states(jobs, tour)

    def ruin(self, solution, jobs):
        """Removes nodes from a solution (inplace).
        Parameters:
            solution: the Solution() class
            jobs: A vector with the nodes (excluding depot) to remove.
        """
        tours_copy = copy.deepcopy(solution.tours)
        tours_to_remove = []

        for tour_copy, tour in zip(tours_copy, solution.tours):
            flag = False
            for x in tour_copy.tour:
                if x in jobs:
                    flag = True
                    tour.tour.remove(x)
            if flag:
                tour.states = []
            if len(tour.tour) == 0:
                tours_to_remove.append(tour)

        for tour in tours_to_remove:
            solution.tours.remove(tour)

        solution.absents = jobs

    def recreate(self, solution, random):
        """Creates a solution."""
        self.create_states(
            solution
        )  # Creates states to match the existing tour. If this doesn't exist, it does nothing.
        jobs = solution.jobs
        vm = solution.vm

        if random:
            rng = np.random.rand()
            random.shuffle(solution.absents, rng)

        remove_from_absents = []

        for x in solution.absents:

            job = jobs[x]
            t, j, delta = self.try_insert(
                solution.tours, job
            )  # Cuando tours está vacío: t=0,j=0, delta=sys.maxsize

            # If the node "job" cannot be inserted anywhere, it creates a new Tour.
            if delta == sys.maxsize:
                v = vm.alloc(job)  # Get vehicle closer to node "job"
                tour = Tour(v)
                self.create_tour_states(jobs, tour)  # (inplace) changes tour.states
                j, delta = self.try_insert_tour(tour, job)
                if delta == sys.maxsize:
                    return False

                self.do_insert(jobs, tour, j, x)
                solution.tours.append(tour)

            else:
                self.do_insert(jobs, solution.tours[t], j, x)

            remove_from_absents.append(x)

        for absent in remove_from_absents:
            solution.absents.remove(absent)  # removes elements from absents.

        self.calc_cost(solution)


class Input:
    def __init__(self, vs, dist_time, cost_per_asent, jobs, depot, temp, c2, sa):
        self.vehicles = vs
        self.dist_time = dist_time
        self.cost_per_absent = cost_per_asent
        self.jobs = jobs
        self.depot = depot
        self.temperature = temp
        self.c2 = c2
        self.sa = sa


class EnvInner:
    def __init__(self, _input, load_solution=False, solution=None):
        _input = _input
        _input.vehicles = _input.vehicles
        _input.jobs = _input.jobs
        _input.dist_time = _input.dist_time

        vm = VehicleManager(_input.vehicles, copy.deepcopy(_input.dist_time))
        self.sol = Solution(_input.jobs, vm)

        self.re = Recreate(copy.deepcopy(_input.dist_time), _input.cost_per_absent)

        if load_solution:
            solution = np.array(solution) - 1
            self.sol.tours = []
            job = self.sol.jobs[4]
            v = self.sol.vm.alloc(job)
            tour = Tour(v)
            for node in solution[1:]:
                if node < 0:
                    self.sol.tours.append(tour)
                    tour = Tour(v)
                else:
                    tour.tour.append(int(node))

            self.sol.absents = []
            self.re.create_states(self.sol)
            self.re.calc_cost(self.sol)

        else:
            self.re.recreate(self.sol, False)

        self.orig_temperature = _input.temperature
        self.temperature = _input.temperature
        self.orig_c = _input.c2
        # self.c starts as whatever it is for random initialization; it later changes
        # to the original one on the call of self.reset_temperature()
        self.c = (FINAL_T / INIT_T) ** (1.0 / RAND_INIT_N_STEPS_SA)
        self.sa = _input.sa

    def states(self):
        states = [copy.deepcopy(tour.states) for tour in self.sol.tours]
        return states

    def tours(self):
        tours = [copy.deepcopy(x.tour) for x in self.sol.tours]
        return tours

    def step(self, jobs):
        """Ruins and reconstructs a solution.
        If self.sa==True (it is), the old solution gets replaced by the new one if :
                the cost of the new solution is better than the old one, or if the new cost < old_cost-temperature*log(rand number),
                where the random number is between [0,1). Then, the new temperature is: t=t*c .
        If self.sa==False, the current solution will be ruin and reconstructed (without checking whether the cost is better or not).

        Parameters:
            jobs: the nodes to remove from the current solution.
        """
        if self.sa:
            cur_sol = copy.deepcopy(self.sol)

            self.re.ruin(cur_sol, jobs)
            self.re.recreate(cur_sol, False)

            rng = np.random.rand()
            new_cost = cur_sol.cost
            old_cost = self.sol.cost

            if new_cost < old_cost or new_cost < old_cost - self.temperature * np.log(
                rng
            ):
                self.sol = cur_sol

            self.temperature = self.temperature * self.c

        else:
            self.re.ruin(self.sol, jobs)
            self.re.recreate(self.sol, False)

    def absents(self):
        return copy.deepcopy(self.sol.absents)

    def cost(self):
        return self.sol.cost

    def reset_temperature(self):
        self.temperature = self.orig_temperature
        self.c = self.orig_c


def input_from_coords(coords):
    jobs = []
    for i, (x, y, demand) in enumerate(coords[1:]):
        jobs.append(Job(i, i + 1, x, y, demand))

    def calc_dist(l, r):
        return ((l[0] - r[0]) ** 2 + (l[1] - r[1]) ** 2) ** 0.5

    dist_time = []
    for x1, y1, _ in coords:
        row = []
        for x2, y2, _ in coords:
            d = calc_dist((x1, y1), (x2, y2))
            row.append(DistTime(d, d))
        dist_time.append(row)

    v = Vehicle(CAP, 0, 0, 1.0, 0, 0.0, 0, 0)

    if N_STEPS > 0:
        alpha_T = (FINAL_T / INIT_T) ** (1.0 / N_STEPS)
        init_temp = 100
    else:
        print("setting temperature to 0")
        alpha_T = 0
        init_temp = 0

    return Input([v], dist_time, 1000, jobs, coords[0][:2], init_temp, alpha_T, True)


class Env(object):
    def __init__(self, n_jobs, input=None, save_best_tours=False):
        self.n_jobs = n_jobs
        assert input != None

        self.input = input
        dist_time = input.dist_time

        # Normalizing distances
        self.dists = np.array([[[x.dist / MAX_DIST] for x in row] for row in dist_time])
        self.save_best_tours = save_best_tours
        self.env = None

    def reset(self, load_solution=False, solution=None):
        self.env = EnvInner(self.input, load_solution, solution)
        self.mapping = {}
        self.cost = 0.0
        self.best = None
        return self.get_states()

    def get_states(self):
        states = (
            self.env.states()
        )  # Vector of the states of each tour ([tour[0].states, tour[1].states,..])
        tours = (
            self.env.tours()
        )  # Vector of the tour of each tour ([tour[0].tour, tour[1].tour,..])
        jobs = self.input.jobs

        nodes = np.zeros((self.n_jobs + 1, 4))
        edges = np.zeros((self.n_jobs + 1, self.n_jobs + 1, 1))
        mapping = {}

        # For creating "nodes", with the info of all the nodes that appear in all the tours. The depot is in "nodes[0]".
        for i, (tour, tour_state) in enumerate(zip(tours, states)):
            for j, (index, s) in enumerate(zip(tour, tour_state[1:])):
                job = jobs[index]
                loc = job.loc

                nodes[loc, :] = [
                    job.weight / CAP,
                    s.weight / CAP,
                    s.dist / MAX_DIST,
                    s.time / MAX_DIST,
                ]
                mapping[loc] = (
                    i,
                    j,
                )  # = (tour_index in tours, index of the element (position) inside the tour)

        # For creating "edges", with the info of whether two nodes are connected:
        # Matrix: (r,c) = 1 if in any tour the vehicle goes from node "r" to node "c".
        for tour in tours:
            edges[0][tour[0] + 1][0] = 1
            for l, r in zip(tour[0:-1], tour[1:]):
                edges[l + 1][r + 1][0] = 1
            edges[tour[-1] + 1][0][0] = 1

        edges = np.stack([self.dists, edges], axis=-1)
        edges = edges.reshape(-1, 2)

        self.mapping = (
            mapping  # Dictionary of the location of each node in the solution.
        )
        self.cost = self.env.cost()  # Cost of the solution.
        if self.best is None or self.cost < self.best:
            self.best = self.cost
            if self.save_best_tours:
                self.best_tours = copy.deepcopy(tours)

        return nodes, edges

    def step(self, to_remove):
        prev_cost = self.cost
        self.env.step(to_remove)

        nodes, edges = self.get_states()
        reward = prev_cost - self.cost
        return nodes, edges, reward

    def reset_temperature(self):
        self.env.reset_temperature()


def create_batch_env(paths, batch_coords, n_jobs, batch_size, save_best_tours=False):
    def step_sub_batch(envs_actions):
        rets = [env.step(act) for env, act in envs_actions]
        rets = list(zip([env for env, _ in envs_actions], rets))
        return rets

    class BatchEnv(object):
        def __init__(
            self, paths, batch_coords, n_jobs, batch_size, save_best_tours=False
        ):
            self.envs = [
                Env(n_jobs, input_from_coords(coords), save_best_tours)
                for coords in batch_coords
            ]

        def reset(self, paths):
            rets = [
                env.reset(load_solution=True, solution=solution)
                for (env, solution) in zip(self.envs, paths)
            ]
            return list(zip(*rets))

        def step(self, actions):
            actions = actions.tolist()
            assert len(actions) == len(self.envs)
            envs_actions = list(zip(self.envs, actions))
            envs, res = list(zip(*step_sub_batch(envs_actions)))
            self.envs = envs
            res = list(zip(*res))
            return res

        def reset_temperature(self):
            for env in self.envs:
                env.reset_temperature()

    return BatchEnv(paths, batch_coords, n_jobs, batch_size, save_best_tours)
