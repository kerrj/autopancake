import numpy as np
import warnings
import matplotlib.pyplot as plt
from scipy import ndimage
import imageio
import argparse

warnings.filterwarnings('ignore', '.*Starting with ImageIO v3 the behavior of this function will switch to that of iio.v3.imread.*')

circumferences = [
	# r=0
	[(0,0)],
	# r=1
	[(1,0),(0,1),(-1,0),(0,-1)],
	# r=2
	[(2,0),(2,1),(1,2),(0,2),(-1,2),(-2,1),(-2,0),(-2,-1),(-1,-2),(0,-2),(1,-2),(2,-1)],
	# r=3
	[(3,0),(3,1),(2,2),(1,3),(0,3),(-1,3),(-2,2),(-3,1),(-3,0),(-3,-1),(-2,-2),(-1,-3),(0,-3),(1,-3),(2,-2),(3,-1)],
	# r=4
	[(4,0),(4,1),(4,2),(3,3),(2,4),(1,4),(0,4),(-1,4),(-2,4),(-3,3),(-4,2),(-4,1),(-4,0),(-4,-1),(-4,-2),(-3,-3),(-2,-4),(-1,-4),(0,-4),(1,-4),(2,-4),(3,-3),(4,-2),(4,-1)],
	# r=5
	[(5,0),(5,1),(5,2),(4,3),(3,4),(2,5),(1,5),(0,5),(-1,5),(-2,5),(-3,4),(-4,3),(-5,2),(-5,1),(-5,0),(-5,-1),(-5,-2),(-4,-3),(-3,-4),(-2,-5),(-1,-5),(0,-5),(1,-5),(2,-5),(3,-4),(4,-3),(5,-2),(5,-1)],
	# r=6
	[(6,0),(6,1),(6,2),(5,3),(5,4),(4,5),(3,5),(2,6),(1,6),(0,6),(-1,6),(-2,6),(-3,5),(-4,5),(-5,4),(-5,3),(-6,2),(-6,1),(-6,0),(-6,-1),(-6,-2),(-5,-3),(-5,-4),(-4,-5),(-3,-5),(-2,-6),(-1,-6),(0,-6),(1,-6),(2,-6),(3,-5),(4,-5),(5,-4),(5,-3),(6,-2),(6,-1)],
	# r=7
	[(7,0),(7,1),(7,2),(6,3),(6,4),(5,5),(4,6),(3,6),(2,7),(1,7),(0,7),(-1,7),(-2,7),(-3,6),(-4,6),(-5,5),(-6,4),(-6,3),(-7,2),(-7,1),(-7,0),(-7,-1),(-7,-2),(-6,-3),(-6,-4),(-5,-5),(-4,-6),(-3,-6),(-2,-7),(-1,-7),(0,-7),(1,-7),(2,-7),(3,-6),(4,-6),(5,-5),(6,-4),(6,-3),(7,-2),(7,-1)],
	# r=8
	[(8,0),(8,1),(8,2),(7,3),(7,4),(6,5),(5,6),(4,7),(3,7),(2,8),(1,8),(0,8),(-1,8),(-2,8),(-3,7),(-4,7),(-5,6),(-6,5),(-7,4),(-7,3),(-8,2),(-8,1),(-8,0),(-8,-1),(-8,-2),(-7,-3),(-7,-4),(-6,-5),(-5,-6),(-4,-7),(-3,-7),(-2,-8),(-1,-8),(0,-8),(1,-8),(2,-8),(3,-7),(4,-7),(5,-6),(6,-5),(7,-4),(7,-3),(8,-2),(8,-1)],
	# r=9
	[(9,0),(9,1),(9,2),(9,3),(8,4),(8,5),(7,6),(6,7),(5,8),(4,8),(3,9),(2,9),(1,9),(0,9),(-1,9),(-2,9),(-3,9),(-4,8),(-5,8),(-6,7),(-7,6),(-8,5),(-8,4),(-9,3),(-9,2),(-9,1),(-9,0),(-9,-1),(-9,-2),(-9,-3),(-8,-4),(-8,-5),(-7,-6),(-6,-7),(-5,-8),(-4,-8),(-3,-9),(-2,-9),(-1,-9),(0,-9),(1,-9),(2,-9),(3,-9),(4,-8),(5,-8),(6,-7),(7,-6),(8,-5),(8,-4),(9,-3),(9,-2),(9,-1)],
	# r=10
	[(10,0),(10,1),(10,2),(10,3),(9,4),(9,5),(8,6),(7,7),(6,8),(5,9),(4,9),(3,10),(2,10),(1,10),(0,10),(-1,10),(-2,10),(-3,10),(-4,9),(-5,9),(-6,8),(-7,7),(-8,6),(-9,5),(-9,4),(-10,3),(-10,2),(-10,1),(-10,0),(-10,-1),(-10,-2),(-10,-3),(-9,-4),(-9,-5),(-8,-6),(-7,-7),(-6,-8),(-5,-9),(-4,-9),(-3,-10),(-2,-10),(-1,-10),(0,-10),(1,-10),(2,-10),(3,-10),(4,-9),(5,-9),(6,-8),(7,-7),(8,-6),(9,-5),(9,-4),(10,-3),(10,-2),(10,-1)]
]


class CircularRange:
	def __init__(self, begin, end, value):
		self.begin, self.end, self.value = begin, end, value

	def __repr__(self):
		return f"[{self.begin},{self.end})->{self.value}"

	def halfway(self):
		return int((self.begin + self.end) / 2)

class Graph:
	class Node:
		def __init__(self, point, index):
			self.x, self.y = point
			self.index = index
			self.connections = {}

		def __repr__(self):
			return f"({self.y},{-self.x})"

		def _addConnection(self, to):
			self.connections[to] = False

		def toDotFormat(self):
			return (f"{self.index} [pos=\"{self.y},{-self.x}!\", label=\"{self.index}\\n{self.x},{self.y}\"]\n" +
				"".join(f"{self.index}--{conn}\n" for conn in self.connections if self.index < conn))


	def __init__(self):
		self.nodes = []

	def __getitem__(self, index):
		return self.nodes[index]

	def __repr__(self):
		return repr(self.nodes)

	def __len__(self):
		return len(self.nodes)


	def addNode(self, point):
		index = len(self.nodes)
		self.nodes.append(Graph.Node(point, index))
		return index

	def addConnection(self, a, b):
		self.nodes[a]._addConnection(b)
		self.nodes[b]._addConnection(a)

	def distance(self, a, b):
		return np.hypot(self[a].x-self[b].x, self[a].y-self[b].y)

	def areConnectedWithin(self, a, b, maxDistance):
		if maxDistance < 0:
			return False
		elif a == b:
			return True
		else:
			for conn in self[a].connections:
				if self.areConnectedWithin(conn, b, maxDistance - self.distance(conn, b)):
					return True
			return False


class GraphBuilder:
	def __init__(self, edges):
		self.edges = edges
		self.ownerNode = np.full(np.shape(edges), -1, dtype=int)
		self.xSize, self.ySize = np.shape(edges)
		self.graph = Graph()

	def getCircularArray(self, center, r, smallerArray = None):
		circumferenceSize = len(circumferences[r])
		circularArray = np.zeros(circumferenceSize, dtype=bool)

		if smallerArray is None:
			smallerArray = np.ones(1, dtype=bool)
		smallerSize = np.shape(smallerArray)[0]
		smallerToCurrentRatio = smallerSize / circumferenceSize

		for i in range(circumferenceSize):
			x = center[0] + circumferences[r][i][0]
			y = center[1] + circumferences[r][i][1]

			if x not in range(self.xSize) or y not in range(self.ySize):
				circularArray[i] = False # consider pixels outside of the image as not-edges
			else:
				iSmaller = i * smallerToCurrentRatio
				a, b = int(np.floor(iSmaller)), int(np.ceil(iSmaller))

				if smallerArray[a] == False and (b not in range(smallerSize) or smallerArray[b] == False):
					circularArray[i] = False # do not take into consideration not connected regions (roughly)
				else:
					circularArray[i] = self.edges[x, y]

		return circularArray

	def toCircularRanges(self, circularArray):
		ranges = []
		circumferenceSize = np.shape(circularArray)[0]

		lastValue, lastValueIndex = circularArray[0], 0
		for i in range(1, circumferenceSize):
			if circularArray[i] != lastValue:
				ranges.append(CircularRange(lastValueIndex, i, lastValue))
				lastValue, lastValueIndex = circularArray[i], i

		ranges.append(CircularRange(lastValueIndex, circumferenceSize, lastValue))
		if len(ranges) > 1 and ranges[-1].value == ranges[0].value:
			ranges[0].begin = ranges[-1].begin - circumferenceSize
			ranges.pop() # the last range is now contained in the first one
		return ranges

	def getNextPoints(self, point):
		"""
		Returns the radius of the circle used to identify the points and
		the points toward which propagate, in a tuple `(radius, [point0, point1, ...])`
		"""

		bestRadius = 0
		circularArray = self.getCircularArray(point, 0)
		allRanges = [self.toCircularRanges(circularArray)]
		for radius in range(1, len(circumferences)):
			circularArray = self.getCircularArray(point, radius, circularArray)
			allRanges.append(self.toCircularRanges(circularArray))
			if len(allRanges[radius]) > len(allRanges[bestRadius]):
				bestRadius = radius
			if len(allRanges[bestRadius]) >= 4 and len(allRanges[-2]) >= len(allRanges[-1]):
				# two consecutive circular arrays with the same or decreasing number>=4 of ranges
				break
			elif len(allRanges[radius]) == 2 and radius > 1:
				edge = 0 if allRanges[radius][0].value == True else 1
				if allRanges[radius][edge].end-allRanges[radius][edge].begin < len(circumferences[radius]) / 4:
					# only two ranges but the edge range is small (1/4 of the circumference)
					if bestRadius == 1:
						bestRadius = 2
					break
			elif len(allRanges[radius]) == 1 and allRanges[radius][0].value == False:
				# this is a point-shaped edge not sorrounded by any edges
				break

		if bestRadius == 0:
			return 0, []

		circularRanges = allRanges[bestRadius]
		points = []
		for circularRange in circularRanges:
			if circularRange.value == True:
				circumferenceIndex = circularRange.halfway()
				x = point[0] + circumferences[bestRadius][circumferenceIndex][0]
				y = point[1] + circumferences[bestRadius][circumferenceIndex][1]

				if x in range(self.xSize) and y in range(self.ySize) and self.ownerNode[x, y] == -1:
					points.append((x,y))

		return bestRadius, points

	def propagate(self, point, currentNodeIndex):
		radius, nextPoints = self.getNextPoints(point)

		# depth first search to set the owner of all reachable connected pixels
		# without an owner and find connected nodes
		allConnectedNodes = set()
		def setSeenDFS(x, y):
			if (x in range(self.xSize) and y in range(self.ySize)
					and np.hypot(x-point[0], y-point[1]) <= radius + 0.5
					and self.edges[x, y] == True and self.ownerNode[x, y] != currentNodeIndex):
				if self.ownerNode[x, y] != -1:
					allConnectedNodes.add(self.ownerNode[x, y])
				self.ownerNode[x, y] = currentNodeIndex # index of just added node
				setSeenDFS(x+1, y)
				setSeenDFS(x-1, y)
				setSeenDFS(x, y+1)
				setSeenDFS(x, y-1)

		self.ownerNode[point] = -1 # reset to allow DFS to start
		setSeenDFS(*point)
		for nodeIndex in allConnectedNodes:
			if not self.graph.areConnectedWithin(currentNodeIndex, nodeIndex, 11):
				self.graph.addConnection(currentNodeIndex, nodeIndex)

		validNextPoints = []
		for nextPoint in nextPoints:
			if self.ownerNode[nextPoint] == currentNodeIndex:
				# only if this point belongs to the current node after the DFS,
				# which means it is reachable and connected
				validNextPoints.append(nextPoint)

		for nextPoint in validNextPoints:
			nodeIndex = self.graph.addNode(nextPoint)
			self.graph.addConnection(currentNodeIndex, nodeIndex)
			self.propagate(nextPoint, nodeIndex)
			self.ownerNode[point] = currentNodeIndex

	def addNodeAndPropagate(self, point):
		nodeIndex = self.graph.addNode(point)
		self.propagate(point, nodeIndex)

	def buildGraph(self):
		for point in np.ndindex(np.shape(self.edges)):
			if self.edges[point] == True and self.ownerNode[point] == -1:
				radius, nextPoints = self.getNextPoints(point)
				if radius == 0:
					self.addNodeAndPropagate(point)
				else:
					for nextPoint in nextPoints:
						if self.ownerNode[nextPoint] == -1:
							self.addNodeAndPropagate(nextPoint)

		return self.graph


def sobel(image):
	image = np.array(image, dtype=float)
	image /= 255.0
	Gx = ndimage.sobel(image, axis=0)
	Gy = ndimage.sobel(image, axis=1)
	res = np.hypot(Gx, Gy)
	res /= np.max(res)
	res = np.array(res * 255, dtype=np.uint8)
	return res[2:-2, 2:-2, 0:3]

def convertToBinaryEdges(edges, threshold):
	result = np.maximum.reduce([edges[:, :, 0], edges[:, :, 1], edges[:, :, 2]]) >= threshold
	if np.shape(edges)[2] > 3:
		result[edges[:, :, 3] < threshold] = False
	return result


def parseArgs(namespace):
	argParser = argparse.ArgumentParser(fromfile_prefix_chars="@",
		description="Image to toolpath")

	argParser.add_argument_group("Data options")
	argParser.add_argument("-i", "--input", type=argparse.FileType('br'), required=True, metavar="FILE",
		help="Input image")
	argParser.add_argument("-o", "--output", type=argparse.FileType('w'), required=True, metavar="FILE",
		help="Output file")
	argParser.add_argument("-e", "--edges", type=str, metavar="MODE",
		help="Consider the input file already as an edges matrix, not as an image of which to detect the edges. MODE should be either `white` or `black`, that is the color of the edges in the image. The image should only be made of white or black pixels.")
	argParser.add_argument("-t", "--threshold", type=int, default=32, metavar="VALUE",
		help="The threshold in range (0,255) above which to consider a pixel as part of an edge (after Sobel was applied to the image or on reading the edges from file with the --edges option)")
	argParser.add_argument("-x", "--max_points", type=int, default=100, metavar="VALUE",
		help="Maximum number of points in the path.")
	argParser.add_argument("-p", "--plot", type=bool, default=False, metavar="VALUE",
		help="Plot results.")

	argParser.parse_args(namespace=namespace)

	if namespace.edges is not None and namespace.edges not in ["white", "black"]:
		argParser.error("mode for --edges should be `white` or `black`")
	if namespace.threshold <= 0 or namespace.threshold >= 255:
		argParser.error("value for --threshold should be in range (0,255)")

def main():
	class Args: pass
	parseArgs(Args)
	image = imageio.imread(Args.input)
	print(f'INPUT IMAGE {image.shape}')
	if Args.edges is None:
		edges = sobel(image)
	elif Args.edges == "black":
		edges = np.invert(image)
	else: # Args.edges == "white"
		edges = image

	edges = convertToBinaryEdges(edges, Args.threshold)
	converter = GraphBuilder(edges)
	converter.buildGraph()
	print(f'NUM NODES IN GRAPH: {len(converter.graph)}')

	xs = [converter.graph[i].x for i in range(len(converter.graph))]
	ys = [converter.graph[i].y for i in range(len(converter.graph))]
	if Args.plot:
		fig, axs = plt.subplots(1, 4)
		axs[0].imshow(image)
		axs[1].imshow(edges)
		axs[2].scatter(xs, ys)
		axs[2].axis('equal')
		axs[3].plot(xs, ys)
		axs[3].axis('equal')
		plt.show()

if __name__ == "__main__":
	main()