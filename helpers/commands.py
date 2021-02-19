import csv


def get_commands_dict(commands_path):
    commands = dict()
    with open(commands_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            commands[row['command']] = row['number']
    return commands