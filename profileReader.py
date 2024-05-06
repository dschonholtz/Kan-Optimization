import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumtime').print_stats(100)  # Adjust the number to show more or fewer lines