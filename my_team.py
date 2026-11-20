# Licensing Information: You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear attribution
# to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# The Pacman AI projects were developed at UC Berkeley. The core projects and
# autograders were created by John DeNero and Dan Klein. Student-side autograding
# was added by Brad Miller, Nick Hay, and Pieter Abbeel.

"""Monte Carlo tree search guided by fixed attack and defense weights."""

import math
import random
from collections import deque
from sys import maxsize
from time import perf_counter

from contest.capture_agents import CaptureAgent
from contest.game import Actions, Directions
from contest.util import Counter


ATTACK_WEIGHTS = {
    'closest_food_dist': -20.0,
    'active_ghost_dist': -10.0,
    'dead_end_risk': -30.0,
    'carrying_risk': -10.0,
    'is_stop': -15.0,
    'bias': -1.5,
    'successor_score': 20.0,
    'ghosts_one_step_away': -30.0,
}

DEFENSE_WEIGHTS = {
    'bias': -1.5,
    'num_invaders': -12.0,
    'distance_from_middle': -8.0,
    'distance_from_closest_invader': -20.0,
    'scared_distance': 16.0,
    'is_home': 12.0,
    'is_stop': -1.5,
}

GAMMA = .8
CARRY_LIMIT = 5
POWER_LIMIT = 5
ENTRY_GUARD_DISTANCE = 3
MCTS_TIME_BUDGET = .15
SEARCH_ITERATIONS = 80000
SEARCH_DEPTH = 5
ROLLOUT_DEPTH = 2
UCT_BALANCER = 30
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
        # different weights if attack or defend
        if role == 'attack':
            self.weights = Counter(ATTACK_WEIGHTS)
        else:
            self.weights = Counter(DEFENSE_WEIGHTS)
        self.gamma = GAMMA
        self.carry_limit = CARRY_LIMIT
        self.power_limit = POWER_LIMIT
        self.mcts_time_budget = MCTS_TIME_BUDGET
    
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

        if self.role == 'attack':
            self.dead_ends = find_dead_ends(walls)
            self.attack_entry = None
        else:
            # Guard the reachable crossing closest to the middle row.
            middle_row = walls.height // 2
            self.guard_position = self.start
            closest_to_middle = math.inf
            for crossing in self.home_boundary:
                rows_from_middle = abs(crossing[1] - middle_row)
                if rows_from_middle < closest_to_middle:
                    if self.get_maze_distance(self.start, crossing) < maxsize:
                        self.guard_position = crossing
                        closest_to_middle = rows_from_middle


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

    def should_return(self, my_ag_state, food):
        #make sure i dont pass the carry limit so that the pacman doesn't stay beeing greedy all the time and returns once it has enough food
        return my_ag_state.num_carrying >= self.carry_limit or (
            my_ag_state.num_carrying > 0 and len(food) <= 2
        )
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

    #we had to add this because the agent would just wait to see if the defending agent left,
    #which obviously never happend, this seemed like the best solution, basically, if a crossing position is defended go to another one
    def update_attack_entry(self, state):
        my_ag_state = state.get_agent_state(self.index)
        # this is only if the attacker has not yet crossed
        if my_ag_state.is_pacman:
            self.attack_entry = None
            return

        ghosts = self.active_ghost_positions(state)
        if not ghosts:
            #keep going to crosss the current crossing if there are no visible ghosts
            # even in the case of a detour you don't want to stop going to the detour just bc there are no visible ghosts.
            return

        position = my_ag_state.get_position()
        enemy_side = 1 if self.red else -1 


        nearest_entry = None
        nearest_distance = maxsize

        safe_entry = None
        safe_distance = maxsize

        #get nearest entry and nearest safe entry, (no ghost there)
        for x, y in self.home_boundary:

            entry = (x + enemy_side, y) #bc home_boundary we stored our side we need to get the other side
            distance = self.get_maze_distance(position, entry)
            if distance < nearest_distance:
                nearest_entry = entry
                nearest_distance = distance
            #check if a visible ghost is far enough to the entry
            if distance < safe_distance and self.nearest_distance(entry, ghosts) > ENTRY_GUARD_DISTANCE:
                safe_entry = entry
                safe_distance = distance


        current_entry = self.attack_entry if self.attack_entry else nearest_entry 
        # if there is nothing to evaluate do nothing
        if current_entry is not None and safe_entry is not None:
            #in the case the current entry is blocked swich to our safe entry
            if self.nearest_distance(current_entry, ghosts) <= ENTRY_GUARD_DISTANCE:
                self.attack_entry = safe_entry

    #there are special cases, so, if the pacman must return bc it has enough pellets or whatever
    #or if the attacker is trying to cross the boundary
    # else it's trying to get food
    def attack_goals(self, my_ag_state, food, returning):
        if returning:
            return self.home_boundary
        if self.attack_entry is not None and not my_ag_state.is_pacman:
            return [self.attack_entry]
        return food


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
        food = self.get_food(state).as_list()
        target_is_home = self.is_home(target)
        returning = self.should_return(my_ag_state, food)

        carried_fraction = my_ag_state.num_carrying / self.carry_limit

        #reward returning food or picking up food
        features['successor_score'] = 0
        if target_is_home:
            features['successor_score'] = carried_fraction
        elif not returning and target in food:
            features['successor_score'] = 1.0

        
        features['closest_food_dist'] = 0
        if not returning:
            goals = self.attack_goals(my_ag_state, food, returning)
            goal_distance = self.nearest_distance(target, goals)
            features['closest_food_dist'] = scaled_distance(goal_distance)

        # Distance from crossing themiddle maters more when returning
        home_distance = self.nearest_distance(target, self.home_boundary)
        home_distance_score = scaled_distance(home_distance)
        if returning:
            features['carrying_risk'] = home_distance_score
        else:
            #when not returning the more pellets u have the less far away from home u wanna be, if u dont have pellet it doesn't really matter how far u are
            features['carrying_risk'] = home_distance_score * (.25 * carried_fraction)

        # the rest of checks are for ghosts so if i have just gotten a capsule or im home i dont need to know
        if target_is_home or target in self.get_capsules(state):
            return

        #find closest gghost
        ghost_positions = self.active_ghost_positions(state)
        closest_ghost_distance = math.inf
        ghosts_one_step_away = 0 #(if 1 step away that is really bad ajja)
        for ghost_position in ghost_positions:
            distance = self.get_maze_distance(target, ghost_position)
            closest_ghost_distance = min(closest_ghost_distance, distance)
            if distance <= 1:
                ghosts_one_step_away += 1

        features['ghosts_one_step_away'] = ghosts_one_step_away
        features['active_ghost_dist'] = max(0, (5 - closest_ghost_distance) / 5)#when ghost gets closer the value needs to be bigger so that it is a worse state

        features['dead_end_risk'] = 0.0
        if not ghost_positions or target not in self.dead_ends:
            return

        if closest_ghost_distance <= 5:
            features['dead_end_risk'] = 1.0
            return

        #dead end be careful
        escape = self.dead_ends[target]
        if escape is not None:

            exit_tile, steps_to_escape = escape
            ghost_distance_to_exit = self.nearest_distance(exit_tile, ghost_positions)


            if ghost_distance_to_exit <= steps_to_escape:
                features['dead_end_risk'] = 1.0
    #defensive heuristic for the proposed destination tile
    def defense_features(self, state, my_ag_state, target, features):

        invaders = self.invaders(state)
        # Count all invaders, but use only visible positions for distances and catches.
        visible_positions = []
        for invader in invaders:
            position = invader.get_position()
            if position is not None:
                visible_positions.append(position)

        target_is_home = self.is_home(target)
        # this feature gives a bonus to moves that stay on our side.
        features['is_home'] = float(target_is_home)
        #predict which invaders we could catch by stepping onto this tile
        catches = 0
        if target_is_home and my_ag_state.scared_timer == 0:
            catches = visible_positions.count(target)
        #Estimate how many invaders would remain after the move.
        features['num_invaders'] = len(invaders) - catches

        if visible_positions:
            distance = self.nearest_distance(target, visible_positions)
            # The scared-distance weight rewards space the chase weight rewards getting close
            if my_ag_state.scared_timer:
                features['scared_distance'] = scaled_distance(distance)
            else:
                features['distance_from_closest_invader'] = scaled_distance(distance)
        else:
            # Return to the central crossing when nobody is visible.
            distance = self.get_maze_distance(target, self.guard_position)
            features['distance_from_middle'] = scaled_distance(distance)

        
    #estimate how promising a move is, seperated the "weights" so that it can be mroe easily tuned
    def heuristic(self, state, action,at_cutoff=False):
        features = self.get_features(state, action)
        if at_cutoff and self.role == 'attack':
            # Reward food progress and pickups in the rounds we actually simulate.
            # Scoring the next pellet here can favor leaving a nearby pellet uneaten.
            features.pop('successor_score', None)
            features.pop('closest_food_dist', None)
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
        #reward it for grabbing a capsule (maybe we should add a negative reward for grabbing a capsule when already having one¿)
        if target in self.get_capsules(state) and target not in self.get_capsules(successor):
            reward += 3
        # if died reward reduce reward, have to give a lower one for attack because when we tested it attack was to afraid to enter the enemy camp
        if respawned:
            if self.role == 'attack':
                return reward - 25
            return reward - 40

        #
        if self.role == 'attack':
            food = self.get_food(state).as_list()
            #get the current goal
            goals = self.attack_goals(before, food, self.should_return(before, food))
            #always go to goal
            direction = 1
        else:
            #list of visible enemy positions
            visible = [enemy.get_position() for enemy in self.invaders(state) if enemy.get_position() is not None]

            
            goals = visible 
            if not goals:
                goals = [self.guard_position]
            # if my defnesive agent are scared don't go towards enemy we dont want to die.
            direction = -1 if (before.scared_timer and visible) else 1

            if self.is_home(target) and before.scared_timer == 0:
                #if it catches an agent positive reward, so if a visible enemy is exactly at the target position
                reward += 15 * visible.count(target)
        
        #as always discourage stopping
        if action == Directions.STOP:
            reward-=0.4


        # if the distance has reduced we should reward
        progress = self.nearest_distance(position, goals) - self.nearest_distance(after.get_position(), goals)
        reward+= direction *progress

        return reward


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
            if index in self.opponents and actor.is_pacman != ours.is_pacman:
                # active ghosts chase Pacman, while Pacman can chase a scared ghost.
                chasing = not actor.scared_timer if not actor.is_pacman else bool(ours.scared_timer)

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
        for ri in range(ROLLOUT_DEPTH):
            actions = self.ordered_actions(state)

            if not actions or perf_counter() >= deadline:
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
        #set a deadline for this turn and check whether the attacker needs a detour.
        deadline = perf_counter() + self.mcts_time_budget
        if self.role == 'attack':
            self.update_attack_entry(state)

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

        if root.untried:
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
