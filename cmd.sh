# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.
:<<!
export train_cmd="slurm.pl --config $1"
export decode_cmd="slurm.pl --config $1"
!
:<<!
export train_cmd="slurm.pl"
export decode_cmd="slurm.pl"
export cuda_cmd="run.pl"
#export cuda_cmd="slurm.pl --mem 10G --gpu 1 --config conf/slurm.conf"
!
#:<<!
export train_cmd="utils/run.pl"
export decode_cmd="utils/run.pl"
export cuda_cmd="utils/run.pl"
#!
export train_nj=30
export decode_nj=10


