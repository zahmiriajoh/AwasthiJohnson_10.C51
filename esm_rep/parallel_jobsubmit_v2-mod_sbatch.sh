####################################################################################################
##############################        SuperCloud Job Submission       ##############################
####################################################################################################
# Author: First Last (email)
# Date: 
#
# Revisions
#    Author (Email)                Date            Reason
#
# Input/Usage
#    command.txt                   File with command to submit as a SuperCloud job
#    job_name                      String to indicate name of job submission
#
#    bash parallel_jobsubmit.sh command.txt job_name
#
# Output files

####################################################################################################
##############################           Code Requirements            ##############################
####################################################################################################
# Scripts
#
# Packages
#
# Other
#    blank_jobsubmit               File with header necessary for SuperCloud job submissions

####################################################################################################
##############################                 Code                   ##############################
####################################################################################################
# 1. Set up variables
    ## Save inputs to variables
    command=$1      ## command.txt
    job_name=$2     ## job_name

    ## Save default file separator to OLD_IFS and set IFS to '\n'
    OLD_IFS=$IFS
    IFS=$'\n'
    
# 2. For each command in command.txt, create and submit a SuperCloud job
# The job iD is saved to job_name_queuelist.txt
    for line in $(cat $command);
        do
            ## Print current command to submit to the terminal
            echo $line

            ## Save the current command to a file job_name_temp_command
            echo $line > $job_name"_temp_command"

            ## Restore default value to file separator using OLD_IFS
            IFS=$OLD_IFS

            ## Create job called job_name_jobsubmit
            cat blank_jobsubmit $job_name"_temp_command" > $job_name"_jobsubmit"

            ## Submit job to SuperCloud and save jobID to job_name_queuelist.txt
            sbatch --job-name="$job_name" $job_name"_jobsubmit" | awk '{print $4 ","}' >> $job_name"_queuelist.txt"            
            ##  Wait 2 sec before processing next command
            sleep 2
        done

# 3. Remove temporary files
    rm $job_name"_temp_command"
    rm $job_name"_jobsubmit"
