# -*- coding: utf-8 -*-

import os
import glob
import re
from PyQt5.QtWidgets import QAction


def findTasks():
    """Parse all files in the present directory to identify
    tasks that should be included in the task menu"""
    path = os.path.dirname(os.path.realpath(__file__))
    files = sorted(glob.glob(path+'/*.py'))
    tasks = []
    for filename in files:
        task = {}
        for line in open(filename, 'r'):
            match = re.search('# MENU'+': (.*)', line)
            if match:
                name = os.path.basename(filename)
                task['name'] = name.split('.')[0]
                title = match.group(1)
                match = re.search('(.*)/(.*)', title)
                if match:
                    task['submenu'] = match.group(1)
                    task['title'] = match.group(2)
                else:
                    task['title'] = title
                continue
            match = re.search('# VISION'+': (.*)', line)
            if match:
                match = re.search('(.*)', match.group(1))
                task['vision'] = eval(match.group(1))
            match = re.search('"""(.*)"""', line)
            if match and 'name' in task:
                task['tip'] = match.group(1)
                break
        if len(task) > 0:
            if 'vision' not in task.keys():
                task['vision'] = False
            tasks.append(task)
    return tasks


def buildTaskMenu(parent):
    """Build menu of available tasks

    For a task task to be included in the menu, it must satisfy
    the following conditions:
    1. source file resides in the tasks directory.
    2. includes comment line with the format:
       MENU: Task Title
       The title "Task Title" will appear in the Task menu,
       and so should be unique.
    3. Optional: if the Task Title is divided by "/", the first
       part will be used to create a sub-menu and the second
       part will be used to create an entry in the sub-menu.
    4. Optional: single-line docstring will appear in the
       top-level status bar as a tool tip that explains the
       tasks' purpose.
    """
    tasks = findTasks()
    if len(tasks) == 0:
        return
    menu = parent.menuTasks
    globals = {'register': parent.tasks.registerTask}
    submenus = dict()
    for task in tasks:
        action = QAction(task['title'], parent)
        if 'tip' in task:
            action.setStatusTip(task['tip'])
        handler = eval(
            'lambda: register("'+task['name']+'", "'+str(task['vision'])+'")', globals)
        action.triggered.connect(handler)
        if 'submenu' in task:
            title = task['submenu']
            if title not in submenus:
                submenus[title] = menu.addMenu(title)
            submenus[title].addAction(action)
        else:
            menu.addAction(action)


if __name__ == '__main__':
    print(findTasks())
