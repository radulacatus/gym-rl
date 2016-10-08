from qlogging import qlogger
from qlogging import experimentsRepository

log = qlogger("testDb")

log.start_experiment({'a': "test",'b': "test"},"test","delete-me")

log.log_score(5.4, 1)

log.log_score(5.2, 2)
log.log_score(1, 3)
log.log_score(0, 4)

log.log_random_hits(12.3,1,5)

log.log_random_hits(1.001,1,7)

id = log.end_experiment()

print id

repo = experimentsRepository("testDb")
print repo.get_full_log(id)