from engine.tournament import Tournament

tournament = Tournament(["greedy", "random", "random_weighted", "random_weighted"])

print(tournament.start(100))
