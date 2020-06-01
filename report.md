# Report

## Learning algorithm

The learning algorithm used is vanilla Deep Q Learning as described in original paper. As an input the vector of state is used instead of an image so convolutional neural nework is replaced with deep neural network. The deep neural network has following layers:

- Fully connected layer - input: 37 (state size) output: 64
- Fully connected layer - input: 64 output 64
- Fully connected layer - input: 64 output: (action size)

Parameters used in DQN algorithm:

- Maximum steps per episode: 1000
- Starting epsilion: 1.0
- Ending epsilion: 0.01
- Epsilion decay rate: 0.995

## Results

![results](plot.png)

```
Episode 100	Average Score: 0.95
Episode 200	Average Score: 3.27
Episode 300	Average Score: 7.50
Episode 400	Average Score: 9.75
Episode 500	Average Score: 12.00
Episode 600	Average Score: 14.10
Episode 700	Average Score: 15.18
Episode 749	Average Score: 16.03
Environment solved in 649 episodes!	Average Score: 16.03
```

The environment got actually solved (average score of 13) in 549 episodes. We trained it little more to score of 16.


## Ideas for future work

1. Extensive hyperparameter optimization
2. Double Deep Q Networks
3. Prioritized Experience Replay
4. Dueling Deep Q Networks
5. RAINBOW Paper
6. Learning from pixels