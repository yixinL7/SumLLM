from transformers import BartPretrainedModel, BartConfig, BartForConditionalGeneration
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder, shift_tokens_right
from transformers.modeling_outputs import Seq2SeqModelOutput
import torch.nn as nn
import torch
from transformers.modeling_outputs import Seq2SeqModelOutput, BaseModelOutput


class CustomBartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.is_scoring_mode = True

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def scoring_mode(self):
        self.is_scoring_mode = True

    def generation_mode(self):
        self.is_scoring_mode = False

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        if self.is_scoring_mode:
            cand_num = decoder_input_ids.size(1)
            encoder_hidden_states = encoder_outputs[0]
            encoder_hidden_states = torch.repeat_interleave(encoder_hidden_states, cand_num, dim=0)
            attention_mask = torch.repeat_interleave(attention_mask, cand_num, dim=0)
            decoder_input_ids = decoder_input_ids.view(-1, decoder_input_ids.size(-1))
            decoder_attention_mask = decoder_attention_mask.view(-1, decoder_attention_mask.size(-1))
        else:
            encoder_hidden_states = encoder_outputs[0]
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )


class BartScorer(BartForConditionalGeneration):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = CustomBartModel(config)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def scoring_mode(self):
        self.model.scoring_mode()
        
    def generation_mode(self):
        self.model.generation_mode()


class BRIO(nn.Module):
    def __init__(self, mname):
        super(BRIO, self).__init__()
        self.model = BartScorer.from_pretrained(mname)
        self.pad_token_id = self.model.config.pad_token_id

    def forward(self, text_id, candidate_id, normalize=True, score_mode="log", length_penalty=1, require_gold=True, adding=0):
        
        batch_size = text_id.size(0)
        
        input_mask = text_id != self.pad_token_id
        cand_mask = candidate_id != self.pad_token_id
        cand_mask[:, :, 0] = 1
        output = self.model(
            input_ids=text_id, 
            attention_mask=input_mask,
            decoder_input_ids=candidate_id, 
            decoder_attention_mask=cand_mask,
            output_hidden_states=False
            )

        output = output[0]  # [bz x cand_num, seq_len, word_dim]
        output = output.view(batch_size, -1, output.size(1), output.size(2)) # [bz, cand_num, seq_len, word_dim]
        probs = output[:, 0]
        output = output[:, :, :-1]  # truncate last token
        candidate_id = candidate_id[:, :, 1:]  # shift right
        cand_mask = candidate_id != self.pad_token_id
        candidate_id = candidate_id.unsqueeze(-1)
        if normalize:
            if score_mode == "log":
                _output = torch.log_softmax(output, dim=3)
            else:
                _output = torch.softmax(output, dim=3)
            scores = torch.gather(_output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        else:
            scores = torch.gather(output, 3, candidate_id).squeeze(-1)  # [bz, cand_num, seq_len]
        cand_mask = cand_mask.float()
        scores = torch.mul(scores, cand_mask).sum(-1) / ((cand_mask.sum(-1) + adding) ** length_penalty) # [bz, cand_num]
        if require_gold:
            output = {'score': scores[:, 1:], "summary_score": scores[:, 0], "probs": probs}
        else:
            output = {'score': scores, "probs": probs}
        return output

    def save_pretrained(self, save_directory):
        self.model.save_pretrained(save_directory)


class label_smoothing_loss(nn.Module):
    def __init__(self, ignore_index, epsilon=0.1, seq_avg=False):
        super(label_smoothing_loss, self).__init__()
        self.ignore_idx = ignore_index
        self.epsilon = epsilon
        self.seq_avg = seq_avg

    def forward(self, input, target):
        input = input.transpose(1, 2) # [batch_size, seq_len, word_num]
        input = torch.log_softmax(input, dim=2)
        k = input.size(2)
        target_prob = torch.ones_like(input).type_as(input) * self.epsilon * 1 / k
        mask = torch.arange(k).unsqueeze(0).unsqueeze(0).expand(target.size(0), target.size(1), -1).type_as(target)
        mask = torch.eq(mask, target.unsqueeze(-1).expand(-1, -1, k))
        target_prob.masked_fill_(mask, 1 - self.epsilon + (self.epsilon * 1 / k))
        loss = - torch.mul(target_prob, input)
        loss = loss.sum(2)
        # mask ignore_idx
        mask = (target != self.ignore_idx).type_as(input)
        if self.seq_avg:
            loss = (torch.mul(loss, mask).sum(1) / mask.sum(1)).mean()
        else:
            loss = (torch.mul(loss, mask).sum() / mask.sum()).mean()
        return loss
    

def AdaptiveRankingLoss(scores, gold_rank, margin, scale=1.0):
    # scores: [batch_size, cand_num]
    # gold_rank: [batch_size, cand_num], larger rank indicates better candidates
    # margin: float, base margin

    cand_num = scores.size(1)

    # Create a mask for the upper triangular part (excluding the diagonal)
    upper_triangular_mask = torch.triu(torch.ones((cand_num, cand_num), dtype=torch.bool), diagonal=1).to(scores.device)

    # Calculate pairwise differences for unique candidate pairs (x-y) only
    pairwise_diffs = scores[:, :, None] - scores[:, None, :]
    pairwise_diffs = pairwise_diffs.masked_select(upper_triangular_mask)

    # Calculate the rank-based distances using gold_rank
    rank_distances = (gold_rank[:, :, None] - gold_rank[:, None, :]).float()
    rank_distances = rank_distances.masked_select(upper_triangular_mask)

    # Create a mask for same gold rank pairs and for rank_distances = 0
    zero_rank_distances_mask = (rank_distances == 0)

    # Calculate adaptive margins depending on rank_distances
    adaptive_margins = (margin + torch.log(rank_distances.masked_select(~zero_rank_distances_mask))) * scale

    # Apply adaptive margins to the pairwise differences
    margin_diffs = adaptive_margins - pairwise_diffs.masked_select(~zero_rank_distances_mask)

    # Calculate the loss using the relu function
    loss = torch.relu(margin_diffs)

    return loss.mean()