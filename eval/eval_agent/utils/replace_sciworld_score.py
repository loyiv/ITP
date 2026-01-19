from scienceworld import ScienceWorldEnv

def step(self, inputStr:str):
    observation = self.server.step(inputStr)
    raw_score = self.server.getScore()
    score = int(round(100 * raw_score))
    completed_success = self.server.getCompleted()

    isCompleted = completed_success
    numMoves = self.getNumMoves()

    reward = score - self.lastStepScore
    self.lastStepScore = score

    if (numMoves > self.envStepLimit):
        isCompleted = True

    if (score < 0):
        isCompleted = True

    infos = {
        'moves': numMoves,
        'raw_score': raw_score,
        'score': score,
        'reward': reward,

        'completed': completed_success,
        'terminal': isCompleted,
        'look': self.look(),
        'inv': self.inventory(),
        'taskDesc': self.taskdescription(),
        'valid': self.getValidActionObjectCombinations(),
        'variationIdx': self.variationIdx,
        'taskName': self.taskName,
        'simplificationStr': self.simplificationStr,
    }

    return observation, reward, isCompleted, infos

def sciworld_monkey_patch():
    ScienceWorldEnv.step = step
    print("Monkey Patched ScienceWorldEnv.step")
