"""
Tree algorithm implemented by Python
"""


def insertion_sort(rand_list):
    rand_list.insert(0, -1)
    for start in range(2, len(rand_list)):
        temp = rand_list[start]
        insert_index = start

        while rand_list[insert_index-1] > temp:
            rand_list[insert_index] = rand_list[insert_index-1]
            insert_index -= 1

        rand_list[insert_index] = temp
    del rand_list[0]
    return rand_list


def bubble_sort(random_list):
    for last in range(len(random_list), 1, -1):
        index = 0
        while index < last - 1:
            if random_list[index] > random_list[index+1]:
                random_list[index+1], random_list[index] = \
                    random_list[index], random_list[index+1]
            index += 1
    return random_list


def selection_sort(random_list):
    for sel in range(0, len(random_list)-1):
        index = sel + 1
        while index < len(random_list):
            if random_list[index] < random_list[sel]:
                random_list[index], random_list[sel] = \
                    random_list[sel], random_list[index]
            index += 1
    return random_list

def shell_sort(random_list):
    def get_h(l):
        h = 1
        while h < l:
            h = h * 3 + 1
        h /= 3
        return int(h)

    h = get_h(len(random_list))
    while h > 0:
        for i in range(h):
            start = h + i
            while start < len(random_list):
                tmp = random_list[start]
                insert_idx = start
                while insert_idx - h > -1 and random_list[insert_idx - h] > tmp:
                    random_list[insert_idx] = random_list[insert_idx - h]
                    insert_idx -= h
                random_list[insert_idx] = tmp
                start += h
        h = int(h/3)

    return random_list


def my_shell_sort(random_list):
    def get_h(l):
        h = 1
        while h < l:
            h = h * 3 + 1
        return int(h/3)

    h = get_h(len(random_list))

    while h > 0:
        for i in range(h):
            start = h + i
            while start < len(random_list):
                tmp = random_list[start]
                ins_idx = start
                while ins_idx > h - 1 and random_list[ins_idx - h] > tmp:
                    random_list[ins_idx] = random_list[ins_idx - h]
                    ins_idx -= h
                random_list[ins_idx] = tmp
                start += h
        h = int(h/3)

    return random_list




















































