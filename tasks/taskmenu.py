# -*- coding: utf-8 -*-

import glob
import re


def findTasks():
    """Parse all files in present directory to identify
    tasks that should be included in the task menu"""
    files = glob.glob('*.py')
    tasks = []
    for filename in files:
        task = {}
        for line in open(filename, 'r'):
            match = re.search('MENU'+': (.*)', line)
            if match:
                task['name'] = filename.split('.')[0]
                task['title'] = match.group(1)
                continue
            match = re.search('"""(.*)"""', line)
            if match:
                if 'name' in task:
                    task['tip'] = match.group(1)
                break
        if len(task) > 0:
            tasks.append(task)
    return tasks

def taskMenu():
    tasks = findTasks()

if __name__ == '__main__':
    print(findTasks())
