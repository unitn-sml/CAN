
from math import ceil, floor
from random import randint, random, seed
import time
from numpy.random import choice
import numpy as np


def leaf(val):

    #Node = namedtuple('Node', ['type', 'children'])
    #return Node(type=val, children=[])
    room_id = val + "_" + str(hash(str(time.time())+val))[-3:]
    return {"id": room_id, "room_type": val}


def node(val, children):
    room_id = val + "_" + str(hash(str(time.time())+val))[-3:]
    #Node = namedtuple('Node', ['type', 'children'])
    #return Node(type=val, children=children)
    return {"id": room_id, "room_type": val, "children": children}


class TreeSampler:

    def __init__(self, seedn, apt_people, res_people, double_res_people,
                 buildings_prob):
        seed(seedn)
        np.random.seed(seedn)
        self.min_apt_people, self.max_apt_people = apt_people
        self.min_res_people, self.max_res_people = res_people
        self.min_double_res_people, self.max_double_res_people\
            = double_res_people
        self.buildings_prob = buildings_prob

    def sampleInstance(self):
        cls = choice(['apt', 'res', 'double_res'], 1, replace=False,
                     p=self.buildings_prob)
        if cls[0] == 'apt':
            nPeople = randint(self.min_apt_people, self.max_apt_people)
            tree = self.sampleApt(nPeople)
        elif cls[0] == 'res':
            nPeople = randint(self.min_res_people, self.max_res_people)
            tree = self.sampleRes(nPeople)
        elif cls[0] == 'double_res':
            nPeople_1 = randint(self.min_double_res_people,
                              self.max_double_res_people)
            nPeople_2 = randint(self.min_double_res_people,
                              self.max_double_res_people)
            tree = self.sampleDoubleRes(nPeople_1, nPeople_2)

        return (cls[0], tree)

    def sampleDoubleRes(self, nPeople_1, nPeople_2):
        '''
        Double Residencees are composed of two dormitory area
         (containing bedrooms and bathrooms only) and two living area
          (containing a kitchen, a living room,
           and possibly additional bathrooms).
        '''

        livingArea_1 = self.sampleResLiving(nPeople_1)
        livingArea_2 = self.sampleResLiving(nPeople_2)
        restingArea_1 = self.sampleResResting(nPeople_1)
        restingArea_2 = self.sampleResResting(nPeople_2)
        res_2 = node("common_room", [livingArea_2] + [restingArea_2])
        return node("common_room", [livingArea_1] + [restingArea_1] +
                    [node("corridor", [res_2])])

    def sampleApt(self, nPeople):
        '''Apartments are composed of a resting area (containing bedrooms and
        bathrooms only) and a living area (containing a kitchen, at
        least a living room, etc.).

        '''
        livingArea = self.sampleAptLiving(nPeople)
        restingArea = self.sampleAptResting(nPeople)
        return node('common_room', [livingArea] + [restingArea])

    def sampleRes(self, nPeople):
        '''Residences are composed of a dormitory area (containing bedrooms
        and bathrooms only) and a living area (containing a kitchen, a
        living room, and possibly additional bathrooms).

        '''
        livingArea = self.sampleResLiving(nPeople)
        restingArea = self.sampleResResting(nPeople)
        return node('common_room', [livingArea] + [restingArea])

    def sampleAptResting(self, nPeople):
        '''Apartments' resting areas are composed of "shared" bathrooms and
        bedrooms, which can possibly include a "private" bathroom.

        '''
        nBedrooms = randint(ceil(nPeople/2.0), nPeople+1)
        nBathrooms = ceil(nPeople/3.0)
        bedrooms = [self.sampleAptBedroom() for _ in range(nBedrooms)]
        bathrooms = [leaf('bathroom') for _ in range(nBathrooms)]
        return node('resting', bedrooms + bathrooms)

    def sampleResResting(self, nPeople):
        '''Residences' resting areas are composed of bedrooms with their
        "private" bathroom.

        '''
        nBedrooms = nPeople
        bedrooms = [node('bedroom', [leaf('bathroom')])
                    for _ in range(nBedrooms)]
        return node('resting', bedrooms)

    def sampleAptBedroom(self):
        hasBathroom = 0.2
        if random() < hasBathroom:
            children = [leaf('bathroom')]
        else:
            children = []

        return node('bedroom', children)

    def sampleAptLiving(self, nPeople):
        '''Apartments' living areas can possibly contain a 'guest' bathroom,
        a number of living rooms and a kitchen.

        '''
        nLivingRooms = randint(ceil(nPeople/4.0), ceil(nPeople/3.0))
        rooms = [leaf('livingroom') for _ in range(nLivingRooms)]
        rooms.append(leaf('kitchen'))

        hasBathroom = 0.1
        if random() < hasBathroom:
            rooms.append(leaf('bathroom'))
        return node('living',rooms)

    def sampleResLiving(self, nPeople):
        '''Residences' living areas always contain a 'guest' bathroom, a lunch
        area composed of living room and kitchen and possibly extra
        living rooms.

        '''
        lunch_area = choice([node('livingroom', [leaf('kitchen')]),
                             node('kitchen', [leaf('livingroom')])])
        rooms = [lunch_area, leaf('bathroom')]
        nExtraRooms = randint(0, floor(nPeople/10.0))

        return node('living', rooms + [leaf('livingroom') for _ in range(nExtraRooms)])

        
        
def getName(node):
    return '%s' % (node.id)
if __name__ == '__main__':
    sampler = TreeSampler(seedn=666, apt_people=[1, 2], res_people=[2, 2],
                          double_res_people=[1,2], buildings_prob=[0.3,0.3,0.4])
    from anytree.importer import DictImporter
    for i in range(10):
        #print(dict(sampler.sampleInstance()[1]._asdict()))
        importer = DictImporter()
        building_type, tree = sampler.sampleInstance()
        tree = importer.import_(tree)
        print(tree)
        from anytree.exporter import DotExporter
        DotExporter(tree, nodenamefunc=getName)\
            .to_picture("tree_{}_{}.png".format(building_type, i))
        print("finish")
