# GPS Spoofing Detection with Parallel Computing

Goal:
Analyze vessel tracking data from AIS records using parallel processing techniques to detect GPS spoofing events. Students will focus on efficient data handling, transformation, and performance evaluation using Python's parallel computing capabilities.

Dataset:
Students will use vessel tracking data from AIS Data: http://web.ais.dk/aisdata/  
Each student must select a single day’s data for analysis.
Each student should pick a different day to avoid overlapping analyses.
Coordinate your chosen date in the chat to ensure unique selections.

## Tasks

1. Parallel Splitting of the Task
   Objective: Strategize the division of AIS data processing into parallelizable sub-tasks.
   Guidance: Discuss different parallel computing techniques, such as data parallelism or task parallelism. Ensure workload balancing among parallel tasks.

2. Implementation of Parallel Processing
   Objective: Develop Python code to process the AIS data in parallel for efficient computation.
   Guidance: Utilize Python libraries for parallel processing.

3. GPS Spoofing Detection
   GPS spoofing is a technique in which someone deliberately manipulates GPS signals, causing receivers to display false location or time data. This can have serious consequences for maritime navigation and security.
   A. Identifying Location Anomalies:
   Detect sudden, unrealistic location jumps that deviate significantly from expected vessel movement patterns.
   B. Analyzing Speed and Course Consistency:
   Identify vessels with inconsistent speed changes or impossible travel distances within a short time.
   C. Comparing Neighboring Vessel Data:
   Check if multiple vessels in the same region report conflicting GPS positions.

4. Evaluating Parallel Processing Efficiency
   Objective: Measure and evaluate the performance efficiency of parallelized processing techniques.
   Guidance:
   Compare execution time between sequential and parallel implementations.
   Use speedup analysis:Speedup=Time (sequential)Time (parallel)\text{Speedup} = \frac{\text{Time (sequential)}}{\text{Time (parallel)}}Speedup=Time (parallel)Time (sequential)
   Analyze CPU and memory usage across different numbers of parallel workers.
   Test different numbers of CPUs and chunk sizes and compare their impact on execution time.
   Visualize results in a graph showing performance improvements across configurations.
   Bonus (+1 point): Students who execute their code on the Vilnius University HPC system and document their experience will receive an additional +1 grade to their task score.

5. Presentation of the Solution
   Objective: Effectively present the solution to the GPS spoofing detection task.
   Guidance: Clearly explain the implemented techniques, performance improvements, and findings in a structured manner.
   Submission and Presentation
   Deadline: All tasks (1-4) must be completed and submitted by the specified end date and time.
   Presentation Format: Up to 5 slides covering solutions to tasks 1-4.
   Selection for Presentation: One student from each group will be randomly chosen to present.
   Requirement: Students must submit their code to a specified code repository (GitHub, GitLab).
