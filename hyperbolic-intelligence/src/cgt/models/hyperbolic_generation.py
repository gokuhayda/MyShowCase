# GERAÇÃO HIPERBÓLICA CORRETA - H-AKORN

import torch
import torch.nn.functional as F

class HLLMChatHyperbolic:
    """Chat com geração hiperbólica correta."""
    
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.conversation = []
        self.model.eval()
        
        # Verificar se modelo tem substrate
        self.has_substrate = hasattr(model, 'substrate') and model.substrate is not None
        
        if self.has_substrate:
            print("✅ Modelo com geometria hiperbólica detectado")
            print("   Usando logits baseados em distância geodésica")
        else:
            print("⚠️  Substrate não encontrado")
            print("   Fallback para geração Euclidiana padrão")
    
    @torch.no_grad()
    def generate_hyperbolic(self, prompt, max_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
        """
        Geração usando distâncias hiperbólicas (CORRETO segundo artigo).
        
        Specification 3.7: logit_i = -d_H(h_final, e_vocab_i) / τ
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        for step in range(max_tokens):
            # Forward pass
            outputs = self.model(input_ids)
            hidden_states = outputs.get('hidden_states')
            
            # Se temos hidden states e substrate, usar método hiperbólico
            if hidden_states is not None and self.has_substrate:
                # Último hidden state: [B, L, D] ou [B, L, D+1]
                h_final = hidden_states[:, -1, :]  # [B, D] ou [B, D+1]
                
                # Vocabulário embeddings
                vocab_embeddings = self.model.embeddings.token_embeddings.weight  # [V, D]
                
                # Se dimensões não batem, usar método Euclidiano
                if h_final.shape[-1] != vocab_embeddings.shape[-1]:
                    # Fallback: usar logits normais
                    logits = outputs['logits'][:, -1, :] / temperature
                else:
                    # MÉTODO CORRETO: Distâncias hiperbólicas
                    try:
                        # Expandir para broadcast: h_final [B, 1, D], vocab [1, V, D]
                        h_expanded = h_final.unsqueeze(1)  # [B, 1, D]
                        v_expanded = vocab_embeddings.unsqueeze(0)  # [1, V, D]
                        
                        # Calcular distâncias geodésicas
                        # d_H(h, e_i) para cada token i no vocabulário
                        distances = self.model.substrate.dist(
                            h_expanded.expand(-1, vocab_embeddings.shape[0], -1).reshape(-1, h_final.shape[-1]),
                            v_expanded.expand(h_final.shape[0], -1, -1).reshape(-1, vocab_embeddings.shape[-1])
                        ).reshape(h_final.shape[0], vocab_embeddings.shape[0])
                        
                        # Logits = -distância / temperatura
                        logits = -distances / temperature
                        
                    except Exception as e:
                        # Se falhar, usar método Euclidiano
                        print(f"⚠️  Hyperbolic generation failed: {e}")
                        logits = outputs['logits'][:, -1, :] / temperature
            else:
                # Sem hidden states ou substrate: método padrão
                logits = outputs['logits'][:, -1, :] / temperature
            
            # Top-k filtering (mesmo para ambos os métodos)
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Top-p (nucleus) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            # Sample (Softmax ainda necessário para normalizar probabilidades)
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            # Stop if EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Decode
        response = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
    
    @torch.no_grad()
    def generate_euclidean(self, prompt, max_tokens=100, temperature=0.8, top_k=50, top_p=0.95):
        """
        Geração Euclidiana padrão (FALLBACK).
        """
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        for _ in range(max_tokens):
            outputs = self.model(input_ids)
            logits = outputs['logits'][:, -1, :] / temperature
            
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        response = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        response = response[len(prompt):].strip()
        
        return response
    
    def generate(self, prompt, **kwargs):
        """
        Auto-select: Hyperbolic if available, Euclidean otherwise.
        """
        if self.has_substrate:
            return self.generate_hyperbolic(prompt, **kwargs)
        else:
            return self.generate_euclidean(prompt, **kwargs)


# COMPARAÇÃO: Euclidiano vs Hiperbólico
def compare_generation_methods(model, tokenizer, device, prompt="Hello"):
    """
    Compara geração Euclidiana vs Hiperbólica.
    """
    print("="*60)
    print("COMPARAÇÃO: Euclidiano vs Hiperbólico")
    print("="*60)
    
    chat = HLLMChatHyperbolic(model, tokenizer, device)
    
    # Método 1: Euclidiano (ATUAL - INCORRETO)
    print("\n1️⃣ MÉTODO EUCLIDIANO (Logits via nn.Linear)")
    print("-" * 60)
    response_euclidean = chat.generate_euclidean(prompt, max_tokens=50, temperature=0.8)
    print(f"Prompt: {prompt}")
    print(f"Response: {response_euclidean}")
    
    # Método 2: Hiperbólico (CORRETO segundo artigo)
    print("\n2️⃣ MÉTODO HIPERBÓLICO (Logits via -d_H)")
    print("-" * 60)
    if chat.has_substrate:
        response_hyperbolic = chat.generate_hyperbolic(prompt, max_tokens=50, temperature=0.8)
        print(f"Prompt: {prompt}")
        print(f"Response: {response_hyperbolic}")
        
        print("\n📊 DIFERENÇA:")
        print(f"   Euclidiano: {len(response_euclidean)} chars")
        print(f"   Hiperbólico: {len(response_hyperbolic)} chars")
        
        if response_euclidean != response_hyperbolic:
            print("   ⚠️  Respostas diferentes (esperado)")
        else:
            print("   ⚠️  Respostas idênticas (suspeito - verificar implementação)")
    else:
        print("   ⚠️  Substrate não disponível - não é possível comparar")
    
    print("\n" + "="*60)
    
    return chat


# EXEMPLO DE USO
"""
# No Colab:
chat_hyperbolic = compare_generation_methods(model, tokenizer, device, "What is")

# Usar na interface
chat_hyperbolic.generate("Explain hyperbolic geometry", max_tokens=100)
"""
