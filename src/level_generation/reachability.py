import numpy as np

'''
Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
USE OR OTHER DEALINGS IN THE SOFTWARE.
'''
# default version, has small bugs
'''
def makeGetNeighbors(jumps, level_as_string, visited, is_solid):
    """
    jumps: list of possible jumps
    level_as_string: the level as list of string of tiles
    visited: a set of already visited points
    is_solid: a map tile -> is solid ?
    """
    # width-1 of the level
    maxX = len(level_as_string[0])-1
    # height-1 of the level
    maxY = len(level_as_string)-1
    jumpDiffs = []
    # jumps from incremental wrt previous pos to absolute. 
    for jump in jumps:
        jumpDiff = [jump[0]]
        for ii in range(1, len(jump)):
            jumpDiff.append((jump[ii][0] - jump[ii-1][0], jump[ii][1] - jump[ii-1][1]))
        jumpDiffs.append(jumpDiff)
    jumps = jumpDiffs

    def getNeighbors(pos):
        # pos: (dist[pos], (x, y, z=-1), 0)
        dist = pos[0] - pos[2]
        pos = pos[1] # (x, y, z)
        visited.add(pos[:2]) 
        # below: tile immediately below
        below = (pos[0],pos[1]+1) # (x, y+1)
        neighbors = []
        # if mario is on a tile without other tiles below, he will fall for sure, returning no neighbors
        if below[1] > maxY:
            return []
        # in this case, pos = (x, y, jump, ii, +-1)
        if pos[2] != -1:
            ii = pos[3] + 1
            jump = pos[2]
            # if jump is not finished yet
            if ii < len(jumps[jump]):
                # multiplying by pos[4] to obtain direction and checking boundaries
                if not (pos[0] + pos[4] * jumps[jump][ii][0] > maxX or 
                        pos[0] + pos[4] * jumps[jump][ii][0] < 0 or 
                        pos[1] + jumps[jump][ii][1] < 0) and \
                        not is_solid(level_as_string[pos[1] + jumps[jump][ii][1]][pos[0] + pos[4] * jumps[jump][ii][0]]):
                    # adding to neighbors if it is not solid and inside boundaries
                    neighbors.append([dist + 1, (pos[0] + pos[4] * jumps[jump][ii][0], pos[1] + jumps[jump][ii][1], jump, ii, pos[4])])
                # 
                if pos[1] + jumps[jump][ii][1] < 0 and not is_solid(level_as_string[pos[1] + jumps[jump][ii][1]][pos[0] + pos[4] * jumps[jump][ii][0]]):
                    neighbors.append([dist+1, (pos[0] + pos[4] * jumps[jump][ii][0], 0, jump, ii, pos[4])])
        
        # is tile under mario is solid, he can jump
        if is_solid(level_as_string[below[1]][below[0]]):
            # if there are other tiles on the right and the tile on the right is not solid
            if pos[0]+1 <= maxX and not is_solid(level_as_string[pos[1]][pos[0]+1]):
                # add tile on the right to set of neighbors
                neighbors.append([dist+1,(pos[0]+1,pos[1],-1)])
            # if there are other tiles on the letf and the tile on the left is not solid
            if pos[0]-1 >= 0 and not is_solid(level_as_string[pos[1]][pos[0]-1]):
                # add tile on the left to set of neighbors
                neighbors.append([dist+1,(pos[0]-1,pos[1],-1)])

            # for all possible jumps, first step from the "earth"
            for jump in range(len(jumps)):
                ii = 0
                # if not ( x + jump[0].x > maxX or y < 0 ) and not tile[y + jump[0].y, x + jump[0].x] is solid // jump to the right
                if not (pos[0]+jumps[jump][ii][0] > maxX or pos[1] < 0) and not is_solid(level_as_string[pos[1]+jumps[jump][ii][1]][pos[0]+jumps[jump][ii][0]]):
                    # add [dist+ii+1, (x + jump[0].x, y + jump[0].y, jump, ii, 1)]
                    neighbors.append([dist+ii+1,(pos[0]+jumps[jump][ii][0],pos[1]+jumps[jump][ii][1],jump,ii,1)]) # 1 for right direction

                # if not ( x - jump[0].x < 0 or y < 0 ) and not tile[y + jump[0].y, x - jump[0].x] is solid // jump to the left
                if not (pos[0]-jumps[jump][ii][0] < 0 or pos[1] < 0) and not is_solid(level_as_string[pos[1]+jumps[jump][ii][1]][pos[0]-jumps[jump][ii][0]]):
                    # add [dist+ii+1, (x - jump[0].x, y + jump[0].y, jump, ii, -1)]
                    neighbors.append([dist+ii+1,(pos[0]-jumps[jump][ii][0],pos[1]+jumps[jump][ii][1],jump,ii,-1)]) # -1 for left direction

        else:
            # falling
            # add [dist+1, (x, y+1, -1)]
            neighbors.append([dist+1,(pos[0],pos[1]+1,-1)])
            # if y + 1 does not exit from the boundaries
            if pos[1] < maxY:
                # if tile bottom-right is not solid, it can be reached
                if not is_solid(level_as_string[pos[1]+1][pos[0]+1]):           # !!! check boundaries
                    # adding to list of neighbors
                    neighbors.append([dist+1.4,(pos[0]+1,pos[1]+1,-1)])
                # if tile bottom-left is not solid, it can be reached
                if not is_solid(level_as_string[pos[1]+1][pos[0]-1]):           # !!! check boundaries
                    # adding to list of neighbors
                    neighbors.append([dist+1.4,(pos[0]-1,pos[1]+1,-1)])
            if pos[1] + 1 < maxY:                                       # completely useless ?
                # if y + 2 is still valid
                if not is_solid(level_as_string[pos[1]+2][pos[0]+1]):
                    neighbors.append([dist+2,(pos[0]+1,pos[1]+2,-1)])
                if not is_solid(level_as_string[pos[1]+2][pos[0]-1]):
                    neighbors.append([dist+2,(pos[0]-1,pos[1]+2,-1)])
        return neighbors
    return getNeighbors
    '''


def check_boundaries(pos, min_bound, max_bound):
    return pos <= max_bound and pos >= min_bound


def get_neighbors(pos, level, jumps):
    """
    Given the coordinates of a tile in a given level, find and return a list of reachable tiles from that position
    using moves and jumps described in the input jumps.
    :param pos: tuple (y, x, jump_index, jump_status, direction). default (y_0, x_0, None, 0, 0)
    :param level: level as a boolean numpy array
    :param jumps: list of possible jumps/moves
    :return: a list of neighbors of the given position
    """

    # width of the level
    neighbors = []
    level_height, level_width = level.shape

    if pos[2] is not None:
        """
        # continue a jump
        """
        jump_status = pos[3] + 1
        jump_index = pos[2]
        # if jump is not finished yet
        if jump_status < len(jumps[jump_index]):
            # multiplying by pos[4] to obtain direction and checking boundaries
            if check_boundaries(pos[1] + pos[4] * jumps[jump_index][jump_status][0], 0, level_width - 1) and \
                    pos[0] + jumps[jump_index][jump_status][1] < level_height and \
                    (pos[0] + jumps[jump_index][jump_status][1] < 0 or not level[pos[0] + jumps[jump_index][jump_status][1], pos[1] + pos[4] * jumps[jump_index][jump_status][0]]):
                neighbors.append((pos[0] + jumps[jump_index][jump_status][1], pos[1] + pos[4] * jumps[jump_index][jump_status][0], jump_index, jump_status, pos[4]))

    if check_boundaries(pos[0], 0, level_height - 2) and check_boundaries(pos[1], 0, level_width - 1) and level[pos[0] + 1, pos[1]]:
        """
        # start a jump or walk left/right. need block below to be solid
        """
        # if there is an other tile on the right and it is not solid
        if pos[1] < level_width - 1 and not level[pos[0], pos[1] + 1]:
            # add tile on the right to set of neighbors
            neighbors.append((pos[0], pos[1] + 1, None))
        # if there is an other tile on the left and it is not solid
        if pos[1] > 0 and not level[pos[0], pos[1] - 1]:
            # add tile on the left to set of neighbors
            neighbors.append((pos[0], pos[1] - 1, None))

        # for all possible jumps, first step from the "earth"
        for jump_index in range(len(jumps)):
            jump_status = 0
            # if not ( x + jump[0].x > maxX or y < 0 ) and not tile[y + jump[0].y, x + jump[0].x] is solid // jump to the right
            if check_boundaries(pos[1] + jumps[jump_index][jump_status][0], 0, level_width - 1) and \
                    not level[pos[0] + jumps[jump_index][jump_status][1], pos[1] + jumps[jump_index][jump_status][0]]:
                # add [dist+ii+1, (x + jump[0].x, y + jump[0].y, jump, ii, 1)]
                # 1 for right direction
                neighbors.append((pos[0] + jumps[jump_index][jump_status][1],
                                  pos[1] + jumps[jump_index][jump_status][0], jump_index, jump_status, 1))

            # if not ( x - jump[0].x < 0 or y < 0 ) and not tile[y + jump[0].y, x - jump[0].x] is solid // jump to the left
            if check_boundaries(pos[1] - jumps[jump_index][jump_status][0], 0, level_width - 1) and \
                    not level[pos[0] + jumps[jump_index][jump_status][1], pos[1] - jumps[jump_index][jump_status][0]]:
                # add [dist+ii+1, (x - jump[0].x, y + jump[0].y, jump, ii, -1)]
                # -1 for left direction
                neighbors.append((pos[0] + jumps[jump_index][jump_status][1],
                                  pos[1] - jumps[jump_index][jump_status][0], jump_index, jump_status, -1))

    elif pos[0] < level_height - 1:
        """
        # keep or start falling
        """
        # add [dist+1, (x, y+1, -1)]
        if pos[0] < -1 or not level[pos[0] + 1, pos[1]]:
            # add block immidiately below
            neighbors.append((pos[0] + 1, pos[1], None))
        # if tile bottom-right is not solid, it can be reached
        if check_boundaries(pos[1] + 1, 0, level_width - 1) and (pos[0] < -1 or not level[pos[0] + 1, pos[1] + 1]):
            # adding to list of neighbors
            neighbors.append((pos[0] + 1, pos[1] + 1, None))
        # if tile bottom-left is not solid, it can be reached
        if check_boundaries(pos[1] - 1, 0, level_width - 1) and (pos[0] < -1 or not level[pos[0] + 1, pos[1] - 1]):
            # adding to list of neighbors
            neighbors.append((pos[0] + 1, pos[1] - 1, None))

    return neighbors


def find_reachable(level, jumps, start_pos=(11, 2)):
    """
    Find the reachability map given a level. The level should be encoded as a bidimensional numpy array of booleans, using True 
    for solid blocks and False for passable blocks.
    :param jumps: list of possible jumps
    :param level: level as a boolean numpy array of air/solid, with shape [h, w] and filled with True (solid) and False (air):
    :param start_pos: optional position to start reachability search. default (11,2)
    :return: a numpy array with the same shape as the input level, with 1 if the corresponding tile is reachable, 0 otherwise.
    """
    
    # modify jumps
    jumpDiffs = []
    for jump in jumps:
        jumpDiff = [jump[0]]
        for jump_index in range(1, len(jump)):
            jumpDiff.append((jump[jump_index][0] - jump[jump_index-1][0], jump[jump_index][1] - jump[jump_index-1][1]))
        jumpDiffs.append(jumpDiff)
    jumps = jumpDiffs

    # create the get neighbour function based on the possible jumps of this level / game
    src = (*start_pos, None, 0, 0)

    reachable = set()
    front = set()
    front.add(src)
    reachable.update(front)

    while len(front) > 0:
        front = {y for x in front for y in get_neighbors(
            x, level, jumps)} - reachable
        reachable.update(front)

    level_reachable = np.zeros(level.shape)
    for r in reachable:
        if r[0] >= 0 and r[0] < level.shape[0] and r[1] >= 0 and r[1] < level.shape[1]:
            level_reachable[r[0], r[1]] = 1

    return level_reachable
