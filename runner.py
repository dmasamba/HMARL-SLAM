import torch
import ray
from model import PolicyNet
from multi_agent_worker import Multi_agent_worker
from parameter import *


class Runner(object):
    def __init__(self, meta_agent_id):
        self.meta_agent_id = meta_agent_id
        self.device = torch.device('cuda') if USE_GPU else torch.device('cpu')
        self.local_network = PolicyNet(LOCAL_NODE_INPUT_DIM, EMBEDDING_DIM)
        self.local_network.to(self.device)

    def get_weights(self):
        return self.local_network.state_dict()
    
    def set_policy_net_weights(self, weights):
        self.local_network.load_state_dict(weights)
    
    def do_job(self, episode_number):
        save_img = True if episode_number % SAVE_IMG_GAP == 0 else False
        save_img = True
        worker = Multi_agent_worker(self.meta_agent_id, self.local_network, episode_number, device=self.device, save_image=save_img)
        worker.run_episode()

        job_results = worker.episode_buffer
        ground_truth_job_results = worker.ground_truth_episode_buffer
        perf_metrics = worker.perf_metrics
        return job_results, ground_truth_job_results, perf_metrics
    
    def job(self, weights_set, episode_number):
        print("starting episode {} on metaAgent {}".format(episode_number, self.meta_agent_id))
        self.set_policy_net_weights(weights_set[0])

        job_results, ground_truth_job_results, metrics = self.do_job(episode_number)

        info = {"id": self.meta_agent_id, "episode_number": episode_number}

        return job_results, ground_truth_job_results, metrics, info


@ray.remote(num_cpus=1, num_gpus=NUM_GPU / NUM_META_AGENT)
class RLRunner(Runner):
    def __init__(self, meta_agent_id):
        super().__init__(meta_agent_id)


if __name__ == '__main__':
    ray.init()
    runner = RLRunner.remote(0)
    job_id = runner.do_job.remote(47)
    out = ray.get(job_id)
    print(out[1])
