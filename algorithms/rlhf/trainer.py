import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import load_dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class RLHFTrainer:
    """
    Fine-tuning de GPT-2 con RLHF (PPO) para generar reviews positivas.
    El reward model es un clasificador de sentimiento preentrenado.
    """

    def __init__(self, lr=1.41e-5, batch_size=16, mini_batch_size=4,
                 ppo_epochs=4, init_kl_coef=0.2, target_kl=6):

        self.device = 0 if torch.cuda.is_available() else -1

        # tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # modelo a entrenar: GPT2 con cabeza de valor para actor-critic
        self.model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')

        # modelo de referencia congelado para la KL penalty
        self.ref_model = AutoModelForCausalLMWithValueHead.from_pretrained('gpt2')

        # clasificador de sentimiento como reward model
        self.sentiment_pipe = pipeline(
            'sentiment-analysis',
            model='distilbert-base-uncased-finetuned-sst-2-english',
            device=self.device
        )

        # configuracion de PPO
        self.config = PPOConfig(
            model_name='gpt2',
            learning_rate=lr,
            batch_size=batch_size,
            mini_batch_size=mini_batch_size,
            gradient_accumulation_steps=1,
            ppo_epochs=ppo_epochs,
            clip_range=0.2,
            init_kl_coef=init_kl_coef,
            target_kl=target_kl,
        )

        self.ppo_trainer = PPOTrainer(
            config=self.config,
            model=self.model,
            ref_model=self.ref_model,
            tokenizer=self.tokenizer,
        )

    def compute_reward(self, texts):
        results = self.sentiment_pipe(texts, truncation=True, max_length=512)
        rewards = []
        for result in results:
            if result['label'] == 'POSITIVE':
                rewards.append(torch.tensor(result['score']))
            else:
                rewards.append(torch.tensor(-result['score']))
        return rewards

    def prepare_dataset(self, n_samples=1000):
        dataset = load_dataset('imdb', split=f'train[:{n_samples}]')

        def tokenize_prompt(sample):
            input_ids = self.tokenizer.encode(
                sample['text'][:100],
                return_tensors='pt',
                truncation=True,
                max_length=32
            )
            return {'input_ids': input_ids.squeeze()}

        dataset = dataset.map(tokenize_prompt)
        dataset.set_format('torch', columns=['input_ids'])
        return dataset

    def train(self, n_epochs=5):
        dataset = self.prepare_dataset()
        rewards_history = []

        generation_kwargs = {
            'min_length': 20,
            'max_new_tokens': 50,
            'top_k': 0,
            'top_p': 1.0,
            'do_sample': True,
            'pad_token_id': self.tokenizer.eos_token_id,
        }

        for epoch in range(n_epochs):
            print(f'\nEpoca {epoch + 1}/{n_epochs}')

            for batch_idx, batch in enumerate(
                DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            ):
                query_tensors = [
                    batch['input_ids'][i] for i in range(len(batch['input_ids']))
                ]

                response_tensors = self.ppo_trainer.generate(
                    query_tensors, **generation_kwargs
                )

                batch_texts = [
                    self.tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                    for r in response_tensors
                ]

                rewards = self.compute_reward(batch_texts)
                avg_reward = torch.stack(rewards).mean().item()
                rewards_history.append(avg_reward)

                self.ppo_trainer.step(query_tensors, response_tensors, rewards)

                if batch_idx % 10 == 0:
                    print(f'  Batch {batch_idx}, reward promedio: {avg_reward:.3f}')
                    print(f'  Ejemplo: {batch_texts[0][:100]}')

        return rewards_history

    def evaluate(self, prompts):
        base_model = GPT2LMHeadModel.from_pretrained('gpt2')
        print("\n=== Modelo base vs RLHF ===\n")

        for prompt in prompts:
            input_ids = self.tokenizer.encode(prompt, return_tensors='pt')

            with torch.no_grad():
                base_output = base_model.generate(
                    input_ids, max_new_tokens=30,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                ft_output = self.model.generate(
                    input_ids, max_new_tokens=30,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            base_text = self.tokenizer.decode(base_output[0], skip_special_tokens=True)