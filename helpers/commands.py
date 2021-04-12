import csv


def get_commands_dict(commands_path):
    commands = dict()
    nums = dict()
    with open(commands_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            commands[row['command']] = row['number']
            nums[row['number']] = row['command']
    return commands, nums

def get_command_by_num(nums, num):
    return nums[num]

def get_commands_list(nums):
    return [get_command_by_num(nums, num) for num in nums]

def get_commands_list_with_silence(nums):
    lst = get_commands_list(nums)
    lst.append('silence')
    return lst

def get_num_by_command(command, commands):
    return commands[command]
