# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear attribution
# to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# The Pacman AI projects were developed at UC Berkeley. The core projects and
# autograders were created by John DeNero and Dan Klein. Student-side autograding
# was added by Brad Miller, Nick Hay, and Pieter Abbeel.

#Monte Carlo tree search guided by fixed attack and defense weights.

import math
import random
from collections import deque
from sys import maxsize
from time import perf_counter

from contest.capture import SIGHT_RANGE
from contest.capture_agents import CaptureAgent
from contest.game import Actions, Directions
from contest.util import Counter, manhattan_distance


ATTACK_WEIGHTS = {
    'closest_food_dist': -20.0,
    'active_ghost_dist': -10.0,
    'dead_end_risk': -30.0,
    'carrying_risk': -10.0,
    'is_stop': -15.0,
    'bias': -1.5,
    'successor_score': 20.0,
    'ghosts_one_step_away': -40.0,
    'capsule_pickup': 20.0,
}

DEFENSE_WEIGHTS = {
    'bias': -1.5,
    'num_invaders': -12.0,
    'distance_from_middle': -8.0,
    'distance_from_closest_invader': -20.0,
    'scared_distance': -16.0,
    'is_home': 12.0,
    'is_stop': -1.5,
}

GAMMA = .8
CARRY_LIMIT = 5
POWER_LIMIT = 5
POWER_CARRY_TIME = 8
ENTRY_GUARD_DISTANCE = 3
ENTRY_SWITCH_MARGIN = 3
LATE_GAME_MOVES = 100
QUIET_DEFENSE_TURNS = 25
MCTS_TIME_BUDGET = .9
SEARCH_ITERATIONS = 80000
SEARCH_DEPTH = 5
ROLLOUT_DEPTH = 2
UCT_BALANCER = 3.0
RANDOM_EXPANSION = False  #true shuffles untried moves. False starts with the best heuristic score
ROLLOUT_RANDOM_CHANCE = .25  # chance to sample from the top moves instead of always picking the best
ROLLOUT_CANDIDATES = 2  #only include x moves in rollout candidates
LEAF_HEURISTIC_WEIGHT = 1.0  #this shoulkd be set to 0 to use only rewards observed in simulations


def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):

    return [
        MCTSAgent(first_index, 'attack'),
        MCTSAgent(second_index, 'defend'),
    ]

#map dead-end tiles to (exit tile, steps to exit), or None if there is no exit
def find_dead_ends(walls):
    #build an "adjacency list" (its a dictionary but still the same concept)
    neighbors = {}
    for tile in walls.as_list(False):
        neighbor_set = set(Actions.get_legal_neighbors(tile, walls))
        #have to remove the current tile
        neighbors[tile] = neighbor_set - {tile}
    # find all tiles that only have 1 neghbour and initialize our queue with those
    dead_ends = deque()
    for tile, tiles_next_to_tile in neighbors.items():
        if len(tiles_next_to_tile) <= 1:
            dead_ends.append(tile)
    
    parents = {}
    while dead_ends:
        tile = dead_ends.popleft()
        # there can only be 1 neighbor/parent bc we always remove the second neighbor of the corridor (*)
        remaining_neighbors = neighbors.pop(tile)
        parent = None

        if remaining_neighbors:
            parent = remaining_neighbors.pop()

        parents[tile] = parent


        if parent is not None:
            #(*) since here we remove the current tile to the parent, when we go pop the parent from dead_ends there will be at most 1 "neighbour"
            neighbors[parent].remove(tile)
            # If it is not 1 then we have left the corridor so we dont add it to dead_ends
            if len(neighbors[parent]) == 1:
                dead_ends.append(parent)

    exits = {}
    
    # we go back bc to know the "exit tile" of the corridor for the furtherst tile we need to know the one for "furthest tile-1"
    for tile, parent in reversed(parents.items()):
        # A parent missing from exits was never removed, so it is the exit itself.
        # A exit_tile of None means there was no exit (it was closed off, i don't know if this can happen but just in case.)
        if parent is None:
            exits[tile] = None
            continue
        #if we still havent processed anything set the distance to 0 (because it is the "preivous" distance and in the last line we will add 1)
        exit_route = (parent,0) 
        if parent in exits:
            exit_route =exits[parent]
        
        if exit_route is None:
            exits[tile] = None
        else:
            exit_tile, distance = exit_route
            exits[tile] = exit_tile, distance + 1
    return exits

#nodes of the MCTS treeS
class SearchNode:
    def __init__(self, state, actions, action=None, reward=0.0):
        self.state = state
        self.untried = actions

        self.action = action
        self.reward = reward
        self.children = []
        self.visits = 0

        self.total = 0.0



    def uct(self, parent_visits):
        exploitation = self.total / self.visits
        exploration = UCT_BALANCER * math.sqrt(math.log(parent_visits) / self.visits)
        return exploitation + exploration


class MCTSAgent(CaptureAgent):
    def __init__(self, index, role):
        super().__init__(index)
        self.role = role
        self.starting_role = role
        # different weights if attack or defend
        if role == 'attack':
            self.weights = Counter(ATTACK_WEIGHTS)
        else:
            self.weights = Counter(DEFENSE_WEIGHTS)
        self.gamma = GAMMA
        self.carry_limit = CARRY_LIMIT
        self.power_limit = POWER_LIMIT
        self.mcts_time_budget = MCTS_TIME_BUDGET
        self.food_cache = None
    
    def register_initial_state(self, game_state):
        super().register_initial_state(game_state) #precomputer maze distances
        #get relevant info like positions of all pacmans and walls and the middle
        self.start = game_state.get_agent_position(self.index)
        self.opponents = self.get_opponents(game_state)

        walls = game_state.get_walls()
        self.middle = walls.width // 2

        left_border = self.middle - 1
        right_border = self.middle
        if self.red:
            home_border = left_border
        else:
            home_border = right_border
        #open tiles in the boundary to be able to cross
        self.home_boundary = []
        for y in range(walls.height):
            if not walls[left_border][y] and not walls[right_border][y]:
                self.home_boundary.append((home_border, y))

        #both agents need these bc they can change roles during the game
        self.home_boundary = [
            tile for tile in self.home_boundary if self.get_maze_distance(self.start, tile) < maxsize
        ]
        self.enemy_entries = [
            (x + (1 if self.red else -1), y) for x, y in self.home_boundary
        ]
        self.dead_ends = find_dead_ends(walls)
        self.attack_entry = None
        self.quiet_turns = 0
        self.food_alarm = None
        self.inferred_invaders = {}
        self.defended_food = set(self.get_food_you_are_defending(game_state).as_list())
        self.home_tiles = [
            tile for tile in walls.as_list(False)
            if self.is_home(tile) and self.get_maze_distance(self.start, tile) < maxsize
        ]
        self.patrol_points = self.find_patrol_points(walls.height // 2)
        self.patrol_index = 0
        if self.starting_role == 'attack':
            self.patrol_index = len(self.patrol_points) - 1
        self.guard_position = self.patrol_points[self.patrol_index]

    def find_patrol_points(self, middle_row):
        if not self.home_boundary:
            return [self.start]

        #count how much food each crossing leads to, then pick crossings in different lanes
        food_counts = Counter()
        for food in self.defended_food:
            crossing = min(self.home_boundary, key=lambda tile: self.get_maze_distance(tile, food))
            food_counts[crossing] += 1

        center = min(self.home_boundary, key=lambda tile: abs(tile[1] - middle_row))
        points = [center]
        lane_size = max(1, len(self.home_boundary) // 3)
        for lane in (self.home_boundary[:lane_size], self.home_boundary[-lane_size:]):
            lane = [tile for tile in lane if tile not in points]
            if lane:
                points.append(max(lane, key=lambda tile: (food_counts[tile], abs(tile[1] - center[1]))))
        return points


    def is_home(self, position):
        return (position[0] < self.middle) == self.red



    #find smallest distance to the target coordinates
    def nearest_distance(self, position, targets):
        nearest = None
        
        for target in targets:

            distance = self.get_maze_distance(position, target)
            if nearest is None or distance < nearest:
                nearest = distance

        if nearest is None:
            return 0
        return nearest

    # get invaders (the ones that are pacmans)
    def invaders(self, state):
        invaders = []
        for index in self.opponents:
            opponent = state.get_agent_state(index)
            if opponent.is_pacman:
                invaders.append(opponent)
        return invaders

    def determine_role(self, state, invaders):
        if state.data.timeleft <= LATE_GAME_MOVES:
            score = self.get_score(state)
            if score > 0:
                return 'defend'
            if score < 0:
                return 'attack'
        if not invaders and self.quiet_turns >= QUIET_DEFENSE_TURNS:
            return 'attack'
        return self.starting_role

    def food_positions(self, state):
        if self.food_cache is None:
            return frozenset(self.get_food(state).as_list())

        #successors share the food grid until someone eats or drops food
        grid_data = state.data.food.data
        key = id(grid_data)
        if key not in self.food_cache:
            food = frozenset(self.get_food(state).as_list())
            self.food_cache[key] = (grid_data, food)
        return self.food_cache[key][1]

    def time_to_return(self, state, position):
        home_distance = self.nearest_distance(position, self.home_boundary)
        return home_distance + 4 >= state.data.timeleft // state.get_num_agents()

    def should_return(self, state, food, ghosts):
        actor = state.get_agent_state(self.index)
        carried = actor.num_carrying
        if not actor.is_pacman or not carried:
            return False
        position = actor.get_position()
        if len(food) <= 2 or self.time_to_return(state, position):
            return True
        if ghosts and self.nearest_distance(position, ghosts) <= (3 if carried == 1 else 5):
            return True

        #keep eating if the enemy ghosts will still be scared for a while
        if carried < self.carry_limit:
            return False
        for index in self.opponents:
            opponent = state.get_agent_state(index)
            if not opponent.is_pacman and opponent.scared_timer <= POWER_CARRY_TIME:
                return True
        return False

    #get positions of enemies you can see
    def active_ghost_positions(self, state):
        positions = []
        for index in self.opponents:
            opponent = state.get_agent_state(index)
            if opponent.is_pacman or opponent.scared_timer >= self.power_limit:
                continue
            position = opponent.get_position()
            if position is not None:
                positions.append(position)
        return positions

    def update_attack_entry(self, state):
        my_ag_state = state.get_agent_state(self.index)
        if my_ag_state.is_pacman:
            self.attack_entry = None
            return

        if not self.enemy_entries:
            return
        ghosts = self.active_ghost_positions(state)
        if self.attack_entry is not None:
            if not ghosts or self.nearest_distance(self.attack_entry, ghosts) > ENTRY_GUARD_DISTANCE:
                return
        position = my_ag_state.get_position()

        def entry_cost(entry):
            cost = self.get_maze_distance(position, entry)
            if ghosts:
                distance = self.nearest_distance(entry, ghosts)
                if distance <= ENTRY_GUARD_DISTANCE:
                    cost += 5 + 5 * (ENTRY_GUARD_DISTANCE + 1 - distance)
                if distance <= 1:
                    cost += 10
            return cost

        #stick with our crossing unless the detour is enough better to be worth turning back
        best_entry = min(self.enemy_entries, key=entry_cost)
        if self.attack_entry is None or entry_cost(best_entry) + ENTRY_SWITCH_MARGIN < entry_cost(self.attack_entry):
            self.attack_entry = best_entry

    #go home with food, go for a capsule under pressure, or keep going towards food
    def attack_goals(self, state, my_ag_state, food, returning, ghosts, capsules):
        position = my_ag_state.get_position()
        if returning and (len(food) <= 2 or self.time_to_return(state, position)):
            return self.home_boundary
        if capsules and ghosts and self.nearest_distance(position, ghosts) <= 5:
            #a nearby capsule is worth going for if we can get there ahead of the ghost
            reachable = []
            for capsule in capsules:
                distance = self.get_maze_distance(position, capsule)
                if distance <= min(6, self.nearest_distance(capsule, ghosts) + 1):
                    reachable.append(capsule)
            if reachable:
                return reachable
        if returning:
            return self.home_boundary
        if self.attack_entry is not None and not my_ag_state.is_pacman:
            return [self.attack_entry]
        return food

    def update_defense_targets(self, state, invaders):
        position = state.get_agent_position(self.index)
        hidden = [index for index in self.opponents
                  if state.get_agent_state(index).is_pacman and state.get_agent_position(index) is None]
        food = set(self.get_food_you_are_defending(state).as_list())
        previous_food = self.defended_food
        if len(self.observation_history) >= 2 and self.get_current_observation() is state:
            previous = self.get_previous_observation()
            previous_food = set(self.get_food_you_are_defending(previous).as_list())
        self.defended_food = food

        #a missing pellet tells us where a hidden invader just was
        stolen = previous_food - food
        if not hidden:
            self.food_alarm = None
        elif stolen:
            self.food_alarm = min(stolen, key=lambda tile: (self.get_maze_distance(position, tile), tile))
        elif self.food_alarm == position:
            self.food_alarm = None

        self.inferred_invaders = {}
        noisy_distances = state.get_agent_distances()
        if hidden and noisy_distances:
            teammates = [state.get_agent_position(index) for index in self.agentsOnTeam]
            unseen_tiles = [tile for tile in self.home_tiles
                            if all(manhattan_distance(tile, teammate) > SIGHT_RANGE for teammate in teammates)]
            for index in hidden:
                possible = [tile for tile in unseen_tiles
                            if state.get_distance_prob(manhattan_distance(position, tile), noisy_distances[index])]
                if possible:
                    #pick a tile near the center of the possible positions to see which lane to check
                    center = (sum(x for x, y in possible) / len(possible),
                              sum(y for x, y in possible) / len(possible))
                    self.inferred_invaders[index] = min(possible, key=lambda tile: manhattan_distance(tile, center))

        if not invaders and position == self.guard_position:
            self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
            self.guard_position = self.patrol_points[self.patrol_index]

    def defense_goals(self, state, actor, visible):
        if actor.is_pacman:
            return self.home_boundary
        hidden = [index for index in self.opponents
                  if state.get_agent_state(index).is_pacman and state.get_agent_position(index) is None]
        if hidden and self.food_alarm is not None:
            return visible + [self.food_alarm]
        inferred = [self.inferred_invaders[index] for index in hidden if index in self.inferred_invaders]
        return visible or inferred or [self.guard_position]


    #collect the numbers used to score one possible move
    def get_features(self, state, action):
     
        my_ag_state = state.get_agent_state(self.index)
        
        target = Actions.get_successor(my_ag_state.get_position(), action)

        #add bias and if an action is not stop is_stop =0,, i have to turn it into float bc the heuristic will multiply it by the weight of stopping 
        features = Counter(bias=1.0, is_stop=float(action == Directions.STOP))
        
        if self.role == 'attack':
            self.attack_features(state, my_ag_state, target, features)
        else:
            self.defense_features(state, my_ag_state, target, features)
        return features

    def attack_features(self, state, my_ag_state, target, features):
        food = self.food_positions(state)
        capsules = self.get_capsules(state)
        ghosts = self.active_ghost_positions(state)
        target_is_home = self.is_home(target)
        returning = self.should_return(state, food, ghosts)
        goals = self.attack_goals(state, my_ag_state, food, returning, ghosts, capsules)
        carried_fraction = my_ag_state.num_carrying / self.carry_limit

        #reward returning food or picking up food
        features['successor_score'] = 0
        if target_is_home:
            features['successor_score'] = carried_fraction
        elif not returning and target in food:
            features['successor_score'] = 1.0
        features['capsule_pickup'] = float(target in capsules)

        features['closest_food_dist'] = 0
        if goals is not self.home_boundary:
            features['closest_food_dist'] = scaled_distance(self.nearest_distance(target, goals))

        #distance from crossing the middle matters more when returning
        home_distance = self.nearest_distance(target, self.home_boundary)
        if goals is self.home_boundary:
            features['carrying_risk'] = scaled_distance(home_distance)
        else:
            features['carrying_risk'] = scaled_distance(home_distance) * (.25 * carried_fraction)

        if target in capsules:
            return

        #a ghost across the border can cut us off even if this move stays on our side
        if ghosts:
            distance = self.nearest_distance(target, ghosts)
            if target_is_home:
                distance = max(2, distance)
            else:
                features['ghosts_one_step_away'] = sum(
                    self.get_maze_distance(target, ghost) <= 1 for ghost in ghosts
                )
            features['active_ghost_dist'] = max(0, (5 - distance) / 5)
        if not target_is_home:
            features['dead_end_risk'] = self.dead_end_risk(state, target, ghosts)

    def dead_end_risk(self, state, target, ghosts):
        if target not in self.dead_ends:
            return 0.0
        escape = self.dead_ends[target]
        if escape is None:
            return 1.0

        #a shallow dead end is fine if we can get to the exit ahead of the ghost
        exit_tile, steps_to_escape = escape
        if ghosts and self.nearest_distance(exit_tile, ghosts) <= steps_to_escape + 1:
            return min(1.0, .2 * steps_to_escape)

        #be a bit careful in deep corridors when we don't know where a ghost is
        if steps_to_escape > 2:
            for index in self.opponents:
                opponent = state.get_agent_state(index)
                if (not opponent.is_pacman and opponent.scared_timer < self.power_limit
                        and opponent.get_position() is None):
                    return min(.2, .035 * (steps_to_escape - 2))
        return 0.0

    #defensive heuristic for the proposed destination tile
    def defense_features(self, state, my_ag_state, target, features):
        invaders = self.invaders(state)
        visible = [invader.get_position() for invader in invaders if invader.get_position() is not None]
        target_is_home = self.is_home(target)
        features['is_home'] = float(target_is_home)
        catches = 0
        if target_is_home and my_ag_state.scared_timer == 0:
            catches = visible.count(target)
        features['num_invaders'] = len(invaders) - catches

        goals = self.defense_goals(state, my_ag_state, visible)
        distance = self.nearest_distance(target, goals)
        if my_ag_state.is_pacman or not invaders:
            features['distance_from_middle'] = scaled_distance(distance)
        elif my_ag_state.scared_timer:
            features['scared_distance'] = scaled_distance(self.shadow_distance(distance))
        else:
            features['distance_from_closest_invader'] = scaled_distance(distance)

    def shadow_distance(self, distance):
        #stay 2 or 3 tiles away so we can chase as soon as the scared timer runs out
        return 2 * max(0, 2 - distance) + max(0, distance - 3)

    #estimate how promising a move is, seperated the "weights" so that it can be mroe easily tuned
    def heuristic(self, state, action, at_cutoff=False):
        features = self.get_features(state, action)
        if at_cutoff and self.role == 'attack':
            #keep the value of being near food, but count the pickup once it actually happens
            features['closest_food_dist'] *= .25
            features['successor_score'] = 0
            features['capsule_pickup'] = 0
        return features * self.weights

    def ordered_actions(self, state, random_order=False):
        #Put the best action last so the search can take it with pop()
        if terminal(state):
            return []
        actions = list(state.get_legal_actions(self.index))
        if random_order:
            random.shuffle(actions)
        else:
            def action_score(action):
                return self.heuristic(state, action)

            actions.sort(key=action_score)
        return actions
   #scores the change between two boards (we added a few heuristic things it's not just the score and that is it)
    def get_reward(self, state, action, successor):
        before = state.get_agent_state(self.index)
        after = successor.get_agent_state(self.index)
        position = before.get_position()
        target = Actions.get_successor(position, action)
        respawned = after.get_position() == self.start and target != self.start
        # i should put all of these "magic numbers" as macros 
        # it should have a high reward for having a high score
    
        reward = 12 * (self.get_score(successor) - self.get_score(state))

        #reward it if carried more than in the last turn, ofc if you lost pellets it subtracts(it is worse to die with pellets than without)
        # shouldn't be punished for returning pellets
        reward += 3 * (after.num_carrying - before.num_carrying + after.num_returned - before.num_returned)
        #a capsule gives us time to eat or get away, so it is worth more than one pellet
        capsules = self.get_capsules(state)
        got_capsule = target in capsules and target not in self.get_capsules(successor)
        if got_capsule:
            reward += 20
        #if we died, keep the penalty higher for defense
        if respawned:
            if self.role == 'attack':
                return reward - 25
            return reward - 40

        after_position = after.get_position()
        if self.role == 'attack':
            food = self.food_positions(state)
            ghosts = self.active_ghost_positions(state)
            returning = self.should_return(state, food, ghosts)
            goals = self.attack_goals(state, before, food, returning, ghosts, capsules)
            progress = self.nearest_distance(position, goals) - self.nearest_distance(after_position, goals)
            if goals is self.home_boundary:
                progress *= 2
            elif not before.is_pacman and not self.is_home(target):
                progress += 2
            if not self.is_home(target) and not got_capsule:
                reward -= 6 * self.dead_end_risk(state, target, ghosts)
        else:
            invaders = self.invaders(state)
            visible = [invader.get_position() for invader in invaders if invader.get_position() is not None]
            goals = self.defense_goals(state, before, visible)
            old_distance = self.nearest_distance(position, goals)
            distance = self.nearest_distance(after_position, goals)
            if before.scared_timer and invaders and not before.is_pacman:
                progress = self.shadow_distance(old_distance) - self.shadow_distance(distance)
            else:
                progress = old_distance - distance
            if self.is_home(target) and before.scared_timer == 0:
                reward += 15 * visible.count(target)
            if not self.is_home(target) and not before.is_pacman:
                reward -= 5

        #as always discourage stopping
        if action == Directions.STOP:
            reward -= .4
        return reward + progress


    #after choosing a testing move we simulate a round for mcts
    def simulate_round(self, state, action):
        #play our move first, then give each other player a turn.
        successor = state.generate_successor(self.index, action)
        for offset in range(1, successor.get_num_agents()):
            if terminal(successor):
                break
            #index back to player 0 after the last player. Imagine we are looking at move player 2 then its 2->3->1->4
            index = (self.index + offset) % successor.get_num_agents()
            actor = successor.get_agent_state(index)
            position = actor.get_position()
            if position is None:
                #a hidden player's turn still uses time and reduces their scared timer.
                actor.scared_timer = max(0, actor.scared_timer - 1)
                successor.data.timeleft -= 1
                continue
            legal = successor.get_legal_actions(index)
            #if i can choose something other than stop choose it bc usually stop i sworse
            moving = [move for move in legal if move != Directions.STOP] or legal
            ours = successor.get_agent_state(self.index)
            #a defender also guards its side when we're about to cross
            guarding_entry = (not actor.is_pacman and not ours.is_pacman
                              and self.get_maze_distance(position, ours.get_position()) <= 6)
            if index in self.opponents and (actor.is_pacman != ours.is_pacman or guarding_entry):
                #active ghosts chase Pacman, while Pacman can chase a scared ghost
                chasing = bool(ours.scared_timer) if actor.is_pacman else not actor.scared_timer
                if guarding_entry:
                    guarding_moves = [move for move in legal
                                      if not self.is_home(Actions.get_successor(position, move))]
                    moving = guarding_moves or moving

                #measure how far this move would leave the other player from us
                def distance_after_move(move):
                    destination = Actions.get_successor(position, move)
                    return self.get_maze_distance(destination, ours.get_position())

                if chasing:
                    step = min(moving, key=distance_after_move)
                else:
                    step = max(moving, key=distance_after_move)
            else:
                #keep moving in the same direction when possible
                heading = actor.get_direction()
                step = heading if heading in moving else moving[0]
            #the next player starts from the board after this move
            successor = successor.generate_successor(index, step)
        return successor
    
    #start with no reward and count the first simulated round at full weight
    #play game till roullout depth
    def rollout(self, state, deadline):
       
        value = 0.0
        discount = 1.0
        for _ in range(ROLLOUT_DEPTH):
            if perf_counter() >= deadline:
                break
            actions = self.ordered_actions(state)
            if not actions:
                break

            #usually take the best move, with a chance to sample from the top moves
            action = actions[-1]

            if random.random() < ROLLOUT_RANDOM_CHANCE:
                action = random.choice(actions[-ROLLOUT_CANDIDATES:])

            successor = self.simulate_round(state, action)
            value += discount * self.get_reward(state, action, successor)
            state = successor

            #count later rewards less each time we advance a round(as always discount actions that are further)
            discount *= self.gamma


        if LEAF_HEURISTIC_WEIGHT == 0 or terminal(state):
            return value
        
        #use the highest heuristic score to estimate the value left on the board
        best_estimate = None
        for action in state.get_legal_actions(self.index):
            estimate = self.heuristic(state, action, at_cutoff=True)

            if best_estimate is None or estimate > best_estimate:
                best_estimate = estimate
        if best_estimate is None:
            best_estimate = 0
        return value + discount * LEAF_HEURISTIC_WEIGHT * best_estimate
    #MAIN FUNCTION, chooses the move
    def choose_action(self, state):
        deadline = perf_counter() + self.mcts_time_budget
        invaders = self.invaders(state)
        self.quiet_turns = 0 if invaders else self.quiet_turns + 1
        role = self.determine_role(state, invaders)
        if role != self.role:
            self.role = role
            weights = ATTACK_WEIGHTS if role == 'attack' else DEFENSE_WEIGHTS
            self.weights = Counter(weights)
            self.attack_entry = None

        #only update our targets from the real board, the search just uses them
        if self.role == 'attack':
            self.update_attack_entry(state)
        else:
            self.update_defense_targets(state, invaders)
        self.food_cache = {}
        try:
            return self.search(state, deadline)
        finally:
            self.food_cache = None

    def search(self, state, deadline):
        actions = self.ordered_actions(state)
        if not actions:
            return Directions.STOP
        #save the best heuristic move in case we can't try every root move in time
        fallback = actions[-1]
        if RANDOM_EXPANSION:

            random.shuffle(actions)
        root = SearchNode(state, actions)

        for _ in range(SEARCH_ITERATIONS):

            if perf_counter() >= deadline:
                break
            path = [root]
            node = root
            #follow UCT until we find an untried move or reach the depth limit.
            while not node.untried and node.children and len(path) <= SEARCH_DEPTH:
                parent_visits = node.visits

                def exploration_score(child):
                    return child.uct(parent_visits)

                node = max(node.children, key=exploration_score)
                path.append(node)

            #try one new move and add its resulting board to the tree.
            if node.untried and len(path) <= SEARCH_DEPTH:
                action = node.untried.pop()
                successor = self.simulate_round(node.state, action)
                reward = self.get_reward(node.state, action, successor)
                
                #the expansion setting decides wether we shuffle the child's moves.
                child_actions = self.ordered_actions(successor, random_order=RANDOM_EXPANSION)
                child = SearchNode(successor, child_actions, action, reward)
                node.children.append(child)
                path.append(child)
                node = child
            value = self.rollout(node.state, deadline)
            #work backward so each node can add its own reward to the later rewards.
            for visited in reversed(path):

                value = visited.reward + self.gamma * value
                visited.total += value
                visited.visits += 1

        if not root.children:
            return fallback

        def visit_priority(child):
            #use average return to break ties in visit count
            av_ret = child.total / child.visits
            return child.visits, av_ret

     
        return max(root.children, key=visit_priority).action

def scaled_distance(distance):
    #make it so one step change th score more when agent is close to its goal
    # like for example distance 20 returns  2/3 but 21 returns 21/31 which is very close to 2/3,
    #instead 1 is 1/11 while 2 is 2/12 which is a bigger distance
    return distance / (distance + 10)


def terminal(state):
    # stop simulating when the game ends or the clock runs out
    return state.is_over() or state.data.timeleft <= 0
