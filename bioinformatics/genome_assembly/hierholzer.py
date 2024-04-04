from random import randint

class Stack:
    def __init__(self):
        self.list = []
    
    def push(self, val):
        self.list.append(val)
    
    def pop(self):
        if self.list == []:
            return None
        
        return self.list.pop()

    def print(self):
        print("stack: ")
        for el in self.list[::-1]:
            print(el, end=" ")
        print()


k = 3
sequence = "TCATTCTTCAGGTCAAA"

kmers = [sequence[i:i + k] for i in range(len(sequence) - k + 1)]
nodes = []

first = True
for kmer in kmers:
    if first:
        nodes.append(kmer[0:k - 1])
        first = False
    nodes.append(kmer[1:k])

dict = {}

for i in range(len(nodes)):
    if i == len(nodes) - 1:
        if nodes[i] not in dict:
            dict[nodes[i]] = [0, 0, []]
    elif nodes[i] in dict:
        dict[nodes[i]][2].append(nodes[i + 1])
    else:
        dict[nodes[i]] = [0, 0, [nodes[i + 1]]]

for key in dict:
    dict[key][2].sort()
    


def build_dict():
    for key, val in dict.items():
        out_count = len(val[2])
        in_count = 0

        for _, val2 in dict.items():
            if key in val2[2]:
                in_count += val2[2].count(key)

        dict[key][0], dict[key][1] = in_count, out_count


def is_eligible(dict):
    def is_all_odd(occurrencies):    
        # test 1: all degrees equal to 0
        return True if occurrencies[0] == 0 and occurrencies[2] == 0 and occurrencies[3] == 0 else False
    
    def is_two_even(occurrencies):
        # test 2: one degree equal to -1, one degree equal to 1
        return True if occurrencies[0] == 1 and occurrencies[2] == 1 and occurrencies[3] == 0 else False

    degrees = [val[1] - val[0] for key, val in dict.items()]        # outgoing_count - ingoing_count
    occurrencies = [degrees.count(-1),      # [count of -1, count of 0, count of 1, count of others]
                    degrees.count(0),
                    degrees.count(1),
                    len(degrees) - (degrees.count(-1) + degrees.count(0) + degrees.count(1))]
    
    return [is_all_odd(occurrencies), is_two_even(occurrencies),
            not is_all_odd(occurrencies) and not is_two_even(occurrencies)]



def __main__():
    build_dict()

    graph_validity = is_eligible(dict)
    
    tpath = Stack()
    epath = Stack()

    if graph_validity[2]:
        print("graph not eligible")
        exit(0)
    else:
        starting_node = None
        ending_node = None
        
        if graph_validity[0]:
            starting_node = dict[randint(0, len(dict))]
            ending_node = dict[randint(0, len(dict))]
        else: 
            for key, val in dict.items():
                if val[1] - val[0] == 1:
                    starting_node = key
                elif val[1] - val[0] == -1:
                    ending_node = key

        tpath.push(starting_node)
        current_node = starting_node

        while True:
            next_node = dict[current_node][2][randint(0, len(dict[current_node][2]) - 1)]
            dict[current_node][1] -= 1
            dict[current_node][2].remove(next_node)
            dict[next_node][0] -= 1

            tpath.push(next_node)

            if next_node == ending_node and len(dict[ending_node][2]) == 0:
                break
            else:
                current_node = next_node

        while True:
            current_node = tpath.pop()
            if current_node == None:
                break
            epath.push(current_node)

            while len(dict[current_node][2]) > 0:
                while True:
                    next_node = dict[current_node][2][randint(0, len(dict[current_node][2]) - 1)]
                    dict[current_node][1] -= 1
                    dict[current_node][2].remove(next_node)
                    dict[next_node][0] -= 1

                    tpath.push(next_node)

                    if len(dict[next_node][2]) == 0:
                        current_node = tpath.pop()
                        epath.push(current_node)
                        break
                    else:
                        current_node = next_node

    epath.print()

    





__main__()