# -*- coding: utf-8 -*-

import glob
import re
from pyqtgraph.Qt import QtGui


def findTasks():
    """Parse all files in the present directory to identify
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
            if match and 'name' in task:
                task['tip'] = match.group(1)
                break
        if len(task) > 0:
            tasks.append(task)
    return tasks


def taskMenu(parent):
    tasks = findTasks()
    if len(tasks) == 0:
        return
    register = parent.instrument.tasks.registerTask
    menu = parent.menuBar().addMenu('&Tasks')
    for task in findTasks():
        action = QtGui.QAction(task['title'], parent)
        action.setStatusTip(task['tip'])
        action.triggered.connect(lambda: register(task['name']))
        menu.addAction(action)


if __name__ == '__main__':
    print(findTasks())
