import random

dogs_avalible = [['dangerous', random.randint(1, 10)], ['calmer', random.randint(1, 10)], ['calm', random.randint(1, 10)]]

class Environment:
    def __init__(self, dogs_avalible):
        self.dogs_avalible = dogs_avalible
        self.total_available = sum([dog[1] for dog in dogs_avalible])

    def update(self):
        self.dogs_avalible = [['dangerous', random.randint(1, 10)],
                              ['calmer', random.randint(1, 10)],
                              ['calm', random.randint(1, 10)]]
        self.total_available = sum([dog[1] for dog in self.dogs_avalible])


class AIAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = {}
        self.learning_rate = 1
        self.discount_factor = 0.95
        self.epsilon = 0.1
        self.health = 10000

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        best_next_q = max([self.get_q_value(next_state, a) for a in range(self.action_size)])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * best_next_q - current_q)
        self.q_table[(state, action)] = new_q

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        else:
            q_values = [self.get_q_value(state, a) for a in range(self.action_size)]
            return q_values.index(max(q_values))

    def take_action(self, dog_type):
        actions = ['try_collect', 'run']
        state = (dog_type,)
        action_index = self.choose_action(state)
        chosen_action = actions[action_index]

        chance = 0
        reward = 0
        damage = 0

        if chosen_action == 'try_collect':
            if dog_type == 'dangerous':
                chance = 0.8
            elif dog_type == 'calmer':
                chance = 0.4
            elif dog_type == 'calm':
                chance = 0.1

            bite_chance = random.uniform(0.0, 1.0)
            if bite_chance < chance:
                if dog_type == 'dangerous':
                    reward = -100
                    damage = 30
                elif dog_type == 'calmer':
                    reward = -5
                    damage = 15
                else:
                    reward = -2
                    damage = 5
                self.health -= damage
            else:
                reward = -50 if dog_type == 'dangerous' else 10

        elif chosen_action == 'run':
            if dog_type == 'dangerous':
                reward = 20
            elif dog_type == 'calmer':
                reward = -10
            elif dog_type == 'calm':
                reward = -20

        self.health = max(0, self.health)

        next_state = (dog_type,)
        self.update_q_value(state, action_index, reward, next_state)

        return chosen_action, reward, damage

def runit(num_iterations, agent=None):
    env = Environment(dogs_avalible)
    if agent is None:
        agent = AIAgent(state_size=2, action_size=2)

    for i in range(num_iterations):
        env.update()
        dog_type_data = random.choice(env.dogs_avalible)
        dog_type = dog_type_data[0]

        action, reward, damage = agent.take_action(dog_type)
        print(f"Iteration {i + 1}: Dog Type: {dog_type}, Action: {action}, Reward: {reward}, Damage: {damage}, Health: {agent.health}")

        if agent.health <= 0:
            print("Agent has no health left. Ending simulation.")
            break

    return agent, i + 1


if __name__ == "__main__":
    num_simulations = 100
    iterations_per_simulation = 1000

    agent = None

    for sim in range(num_simulations):
        print(f"\n--- Starting simulation {sim + 1} ---")

        agent, iterations = runit(iterations_per_simulation, agent)

        print(f"\nSimulation {sim + 1} completed after {iterations} iterations.")
        print(f"Final health: {agent.health}")

        # Print Q-values for each dog type
        for dog_type in ['dangerous', 'calmer', 'calm']:
            print(f"\nQ-values for {dog_type} dogs:")
            for action in range(agent.action_size):
                q_value = agent.get_q_value((dog_type,), action)
                action_name = 'try_collect' if action == 0 else 'run'
                print(f"Action: {action_name}, Q-value: {q_value:.2f}")

        agent.health = 10000

    print("\nTraining completed.")
