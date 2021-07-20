import argparse


class Node():

    def __init__(self, state, parent, action, heuristic):
        self.state = state
        self.parent = parent
        self.action = action
        if heuristic:
            self.heuristic = heuristic


class StackFrontier():

    def __init__(self):
        self.frontier = []

    def add(self, node):
        self.frontier.append(node)

    def contains_state(self, state):
        return any(node.state == state for node in self.frontier)

    def empty(self):
        return len(self.frontier) == 0

    def remove(self):
        if not self.empty():
            return self.frontier.pop()
        else:
            raise Exception("Empty Frontier")


class QueueFrontier(StackFrontier):

    def remove(self):
        if not self.empty():
            return self.frontier.pop(0)
        else:
            raise Exception("Empty Frontier")


class GreedyFrontier(StackFrontier):

    def remove(self):
        if not self.empty():
            hmin = self.frontier[0].heuristic
            best_choice = self.frontier[0]
            for node in frontier:
                if node.heuristic < hmin:
                    hmin = node.heuristic
                    best_choice = node

            return best_choice
        else:
            raise Exception("Empty Frontier")


class Maze():

    def __init__(self, filename):
        ''' Reads file and finds Walls. '''
        with open(filename) as f:
            contents = f.read()

        # Check for start and end
        if contents.count('A') != 1:
            raise Exception("Maze must have exactly 1 start point.")
        if contents.count('B') != 1:
            raise Exception("Maze must have exactly 1 goal point.")

        # Find height and width of maze
        contents = contents.splitlines()
        self.height = len(contents)
        self.width = max(len(line) for line in contents)

        # Find walls
        self.walls = []
        for i in range(self.height):
            row = []
            for j in range(self.width):
                try:
                    if contents[i][j] == 'A':
                        self.start = (i, j)
                        row.append(False)
                    elif contents[i][j] == 'B':
                        self.goal = (i, j)
                        row.append(False)
                    elif contents[i][j] == ' ':
                        row.append(False)
                    else:
                        row.append(True)
                except IndexError:
                    row.append(False)
            self.walls.append(row)

        self.solution = None

    def print_maze(self):
        ''' Prints maze to the terminal. '''
        solution = self.solution[1] if self.solution else None

        print() 
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):
                if col:
                    print('█', end='')
                elif (i, j) == self.start:
                    print('A', end='')
                elif (i, j) == self.goal:
                    print('B', end='')
                elif solution is not None and (i, j) in solution:
                    print('*', end='')
                else:
                    print(' ', end='')
            print()
        print()

    def neighbours(self, state):
        ''' Finds neighbours reachable through actions of a particular state'''
        row, col = state
        candidates = [
            ('up', (row - 1, col)),
            ('down', (row + 1, col)),
            ('left', (row, col - 1)),
            ('right', (row, col + 1))
        ]

        result = []
        for action, (r, c) in candidates:
            if 0 <= r < self.height and 0 <= c < self.width and not self.walls[r][c]:
                result.append((action, (r, c)))
        return result

    def algorithm(self, method):
        if method == 'dfs':
            frontier = StackFrontier()
            heuristic = None
        elif method == 'bfs':
            frontier = QueueFrontier()
            heuristic = None
        elif method == 'gbfs':
            frontier = GreedyFrontier()
            heuristic = self.manhattan_dist(self.start)
        elif method == 'gbfs':
            frontier = GreedyFrontier()
            heuristic = self.manhattan_dist(self.start) + self.cumulative_cost(self.start)
        else:
            raise Exception("Method not recognised")
        return frontier, heuristic

    def solve(self, method):
        ''' Finds solution, if it exists. '''

        # Keep track of no. of states explored
        self.num_explored = 0

        # Initialize the frontier to just the starting position
        frontier, h = self.algorithm(method)
        start = Node(state=self.start, parent=None, action=None, heuristic=h)
        frontier.add(start)

        # Initialise an empty explored set
        self.explored = set()

        # Keep looping until solution found
        while True:

            # If nothing is left in frontier, then no path
            if frontier.empty():
                raise Exception("No Solution")

            # Choose a node from frontier
            node = frontier.remove()
            self.num_explored +=1

            # If node is goal, then we have a Solution
            if node.state == self.goal:
                actions = []
                path = []
                while node.parent is not None:
                    actions.append(node.action)
                    path.append(node.state)
                    node = node.parent
                actions.reverse()
                path.reverse()
                self.solution = (action, path)
                return

            # Mark node as explored
            self.explored.add(node.state)

            # Add neighbour to frontier
            for action, state in self.neighbours(node.state):
                if not frontier.contains_state(state) and state not in self.explored:
                    child = Node(state=state, parent=node, action=action, heuristic)
                    frontier.add(child)

    def output_image(self, filename, show_solution=True, show_explored=False):
        ''' Uses the Pillow library to create an image of the maze. '''
        from PIL import Image, ImageDraw
        cell_size = 50
        cell_border = 2

        # Create blank canvas
        size = (self.width * cell_size, self.height * cell_size)
        img = Image.new("RGBA", size, "black")
        draw = ImageDraw.Draw(img)

        solution = self.solution[1] if self.solution is not None else None
        for i, row in enumerate(self.walls):
            for j, col in enumerate(row):

                # Walls
                if col:
                    fill = (40, 40, 40)

                # Start
                elif (i, j) == self.start:
                    fill = (255, 0, 0)

                # Goal
                elif (i, j) == self.goal:
                    fill = (0, 171, 28)

                # Solution
                elif solution is not None and show_solution and (i, j) in solution:
                    fill = (220, 235, 113)

                # Explored
                elif solution is not None and show_explored and (i, j) in self.explored:
                    fill = (212, 97, 85)

                # Empty cell
                else:
                    fill = (237, 240, 252)

                # Draw cell
                p1 = (j * cell_size + cell_border, i * cell_size + cell_border)
                p2 = ((j + 1) * cell_size - cell_border, (i + 1) * cell_size - cell_border)
                draw.rectangle(([p1, p2]), fill=fill)

        img.show(filename)


parser = argparse.ArgumentParser()
parser.add_argument("mazefile", help="file containing maze in .txt format")
args = parser.parse_args()

m = Maze(args.mazefile)
print("Maze:")
m.print_maze()
print("Solving...")
m.solve()
print("States Explored:", m.num_explored)
print("Solution:")
m.print_maze()
m.output_image("maze.png", show_explored=True)
