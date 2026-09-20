import torch
from loguru import logger

def log_input_data_details(sample_input_ids, inputs, tokenizer, current_epoch, step_in_epoch, global_step, dataset_type="single"):
    """
    Log detokenized input data for debugging and monitoring.
    
    Args:
        sample_input_ids: The input_ids tensor for one sample
        inputs: The full batch of inputs
        tokenizer: The tokenizer for detokenization
        current_epoch: Current training epoch
        step_in_epoch: Current step within the epoch
        global_step: Global training step
        dataset_type: Type of dataset ("with_refusal", "without_refusal", or "single")
    """
    try:
        detokenized_text = tokenizer.decode(sample_input_ids, skip_special_tokens=False)
        
        logger.info(f"=== EPOCH {current_epoch}, STEP {step_in_epoch} (Global Step {global_step}) - {dataset_type.upper()} DATASET ===")
        logger.info(f"Sample input_ids shape: {sample_input_ids.shape}")
        logger.info(f"Detokenized text (first sample):")
        logger.info(f"{detokenized_text}")
        
        # Log masked versions if masks are available
        # Check for labels (often used for loss masking in language modeling)
        if 'labels' in inputs:
            labels = inputs['labels'][0]
            
            try:
                # In language modeling, labels often have -100 for tokens that don't contribute to loss
                ignored_label_mask = labels == -100
                valid_label_mask = labels != -100
                
                # Show the ignored/masked tokens first (the ones that get ignored)
                if ignored_label_mask.any():
                    ignored_tokens = sample_input_ids[ignored_label_mask]
                    ignored_text = tokenizer.decode(ignored_tokens, skip_special_tokens=False)
                    logger.info(f"--- IGNORED/MASKED TOKENS (labels == -100) ---")
                    logger.info(f"Labels shape: {labels.shape}")
                    logger.info(f"Number of ignored tokens: {ignored_label_mask.sum().item()}/{len(labels)}")
                    logger.info(f"Ignored tokens text:")
                    logger.info(f"{ignored_text}")
                else:
                    logger.info(f"--- NO IGNORED TOKENS (no labels are -100) ---")
                
                # Also show loss contributing tokens for comparison
                if valid_label_mask.any():
                    # Show only tokens that contribute to the loss
                    loss_contributing_tokens = sample_input_ids[valid_label_mask]
                    loss_contributing_labels = labels[valid_label_mask]
                    
                    loss_text = tokenizer.decode(loss_contributing_tokens, skip_special_tokens=False)
                    logger.info(f"--- LOSS CONTRIBUTING TOKENS (labels != -100) ---")
                    logger.info(f"Number of loss-contributing tokens: {valid_label_mask.sum().item()}/{len(labels)}")
                    logger.info(f"Loss-contributing text:")
                    logger.info(f"{loss_text}")
                    
                    # Show the actual label values for these tokens
                    label_text = tokenizer.decode(loss_contributing_labels, skip_special_tokens=False)
                    logger.info(f"Corresponding labels:")
                    logger.info(f"{label_text}")
                else:
                    logger.info(f"--- NO LOSS CONTRIBUTING TOKENS (all labels are -100) ---")
                
            except Exception as e:
                logger.error(f"Error processing labels mask at epoch {current_epoch}, step {step_in_epoch}: {e}")
        
        # Check for any other custom mask fields
        mask_fields = [key for key in inputs.keys() if 'mask' in key.lower() and key not in ['attention_mask']]
        for mask_field in mask_fields:
            try:
                mask = inputs[mask_field][0]
                if mask.dtype == torch.bool or (mask.dtype in [torch.int, torch.long] and set(mask.unique().tolist()).issubset({0, 1})):
                    # This looks like a binary mask
                    masked_tokens = sample_input_ids[mask == 1]
                    if len(masked_tokens) > 0:
                        masked_text = tokenizer.decode(masked_tokens, skip_special_tokens=False)
                        logger.info(f"--- CUSTOM MASK: {mask_field.upper()} ---")
                        logger.info(f"Mask shape: {mask.shape}")
                        logger.info(f"Number of selected tokens: {mask.sum().item()}/{len(mask)}")
                        logger.info(f"Selected tokens text:")
                        logger.info(f"{masked_text}")
                        
            except Exception as e:
                logger.error(f"Error processing custom mask '{mask_field}' at epoch {current_epoch}, step {step_in_epoch}: {e}")
        
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error detokenizing input data at epoch {current_epoch}, step {step_in_epoch}: {e}")


def log_token_level_losses(
    sample_input_ids,
    token_losses,
    inputs,
    tokenizer,
    current_epoch,
    step_in_epoch,
    global_step,
    dataset_type="harmful",
    loss_type="harmful",
    second_tokens_first_sample=None,
    selected_positions=None,  # positions of tokens selected by diff mask (1D tensor or list)
    perturbed_token_losses=None,  # tensor of perturbed model losses for the first sample (same shape as token_losses)
    diff_token_losses=None,  # tensor of (harmful-perturbed) losses for the first sample
):
    """
    Log token-level losses for the first sample in a batch.
    
    Args:
        sample_input_ids: The input_ids tensor for one sample [seq_len]
        token_losses: Per-token loss values for the sample [seq_len-1] (shifted for next-token prediction)
        inputs: The full batch of inputs
        tokenizer: The tokenizer for detokenization
        current_epoch: Current training epoch
        step_in_epoch: Current step within the epoch
        global_step: Global training step
        dataset_type: Type of dataset ("harmful", "safe", etc.)
        loss_type: Type of loss being logged ("harmful", "perturbed", etc.)
        second_tokens_first_sample: Tensor of shape [seq_len-1] containing the
            ID of the model's second-highest-probability prediction at each
            next-token position (optional).  Only used for richer logging.
    """
    try:
        logger.info(f"=== TOKEN-LEVEL {loss_type.upper()} LOSSES - EPOCH {current_epoch}, STEP {step_in_epoch} (Global Step {global_step}) - {dataset_type.upper()} ===")
        
        # Get labels if available for proper alignment
        labels = inputs.get('labels', None)
        if labels is not None:
            # Ensure the label tensor is on CPU to match indices generated
            labels_first_sample = labels[0].detach().cpu()
            
            # For next-token prediction, we need to align input_ids with shifted labels
            # input_ids: [0, 1, 2, 3, 4] -> predicting tokens [1, 2, 3, 4]
            # labels:    [1, 2, 3, 4, -100] (typically)
            # losses:    [loss_for_1, loss_for_2, loss_for_3, loss_for_4]
            
            # Also ensure input tokens are on CPU
            input_tokens = sample_input_ids[:-1].cpu()  # Remove last token since we don't predict beyond it
            target_tokens = labels_first_sample[1:]  # Remove first token since it's not predicted
            
            # Filter out ignored positions (-100 labels)
            IGNORE_INDEX = -100
            valid_mask = target_tokens != IGNORE_INDEX
            
            if valid_mask.any():
                # Extract second-highest predictions if provided
                if second_tokens_first_sample is not None:
                    second_tokens_full = second_tokens_first_sample.cpu()
                    second_tokens_valid = second_tokens_full[valid_mask]
                else:
                    second_tokens_valid = None
                
                valid_input_tokens = input_tokens[valid_mask]
                valid_target_tokens = target_tokens[valid_mask]
                valid_losses = token_losses[valid_mask]
                
                logger.info(f"Number of loss-contributing tokens: {valid_mask.sum().item()}")
                logger.info(f"Token losses shape: {token_losses.shape}")
                logger.info(f"Valid losses shape: {valid_losses.shape}")
                
                # ======================= FULL SENTENCE WITH LOSSES =======================
                logger.info(f"--- FULL SENTENCE WITH TOKEN LOSSES ---")
                
                # Create a visualization of the full sentence with losses
                sentence_with_losses = []
                for i, (input_token, target_token, loss_value) in enumerate(zip(valid_input_tokens, valid_target_tokens, valid_losses)):
                    # Decode the target token (what we're predicting)
                    target_text = tokenizer.decode([target_token], skip_special_tokens=False)
                    # Clean up the token text (remove spaces, special chars for display)
                    clean_target = target_text.replace('▁', ' ').replace('<0x0A>', '\\n').strip()
                    if clean_target == '':
                        clean_target = f'[{target_token.item()}]'  # Fallback to token ID if empty
                    
                    # Build token(loss) string (without second-best)
                    token_with_loss = f"{clean_target}({loss_value:.3f})"
                    sentence_with_losses.append(token_with_loss)
                
                # Join tokens with spaces and display
                full_sentence = ' '.join(sentence_with_losses)
                logger.info(f"Sentence: {full_sentence}")
                
                # Also show a cleaner version without loss values for readability
                clean_tokens = []
                second_tokens_clean = []  # track second-best separately
                for idx_t, target_token in enumerate(valid_target_tokens):
                    target_text = tokenizer.decode([target_token], skip_special_tokens=False)
                    clean_target = target_text.replace('▁', ' ').replace('<0x0A>', '\n').strip()
                    if clean_target == '':
                        clean_target = f'[{target_token.item()}]'
                    clean_tokens.append(clean_target)

                    if second_tokens_valid is not None:
                        alt_tok_id = second_tokens_valid[idx_t]
                        alt_tok_text = tokenizer.decode([alt_tok_id], skip_special_tokens=False).replace('▁', ' ').strip()
                        if alt_tok_text == '':
                            alt_tok_text = f'[{alt_tok_id.item()}]'
                        second_tokens_clean.append(f"{clean_target}->{alt_tok_text}")
                
                clean_sentence = ' '.join(clean_tokens)
                logger.info(f"Clean text: {clean_sentence}")

                # If second-best predictions are available, log them separately
                if second_tokens_valid is not None and len(second_tokens_clean) > 0:
                    second_sent = ' '.join(second_tokens_clean)
                    logger.info(f"Second-best predictions: {second_sent}")
                # =====================================================================
                
                # Log top-k highest and lowest losses
                k = min(10, len(valid_losses))  # Log up to 10 tokens
                
                # Get indices of highest and lowest losses
                _, top_indices = torch.topk(valid_losses, k)
                _, bottom_indices = torch.topk(valid_losses, k, largest=False)
                
                logger.info(f"--- TOP {k} HIGHEST LOSSES ---")
                for i, idx in enumerate(top_indices):
                    input_token = valid_input_tokens[idx]
                    target_token = valid_target_tokens[idx]
                    loss_value = valid_losses[idx].item()
                    
                    # Decode individual tokens
                    input_text = tokenizer.decode([input_token], skip_special_tokens=False)
                    target_text = tokenizer.decode([target_token], skip_special_tokens=False)
                    
                    logger.info(f"  {i+1}. Loss: {loss_value:.4f} | Input: '{input_text}' -> Target: '{target_text}' | IDs: {input_token.item()} -> {target_token.item()}")
                
                logger.info(f"--- TOP {k} LOWEST LOSSES ---")
                for i, idx in enumerate(bottom_indices):
                    input_token = valid_input_tokens[idx]
                    target_token = valid_target_tokens[idx]
                    loss_value = valid_losses[idx].item()
                    
                    # Decode individual tokens
                    input_text = tokenizer.decode([input_token], skip_special_tokens=False)
                    target_text = tokenizer.decode([target_token], skip_special_tokens=False)
                    
                    logger.info(f"  {i+1}. Loss: {loss_value:.4f} | Input: '{input_text}' -> Target: '{target_text}' | IDs: {input_token.item()} -> {target_token.item()}")
                
                # Log summary statistics
                mean_loss = valid_losses.mean().item()
                std_loss = valid_losses.std().item()
                max_loss = valid_losses.max().item()
                min_loss = valid_losses.min().item()
                
                logger.info(f"--- LOSS STATISTICS ---")
                logger.info(f"Mean: {mean_loss:.4f}, Std: {std_loss:.4f}, Max: {max_loss:.4f}, Min: {min_loss:.4f}")

                # ---------------- LOG SELECTED DIFF TOKENS ----------------
                if (
                    selected_positions is not None
                    and diff_token_losses is not None
                    and perturbed_token_losses is not None
                ):
                    try:
                        sel_pos_cpu = (
                            selected_positions
                            if isinstance(selected_positions, list)
                            else selected_positions.detach().cpu()
                        )
                        if sel_pos_cpu.numel() > 0:
                            logger.info("--- SELECTED DIFF TOKENS ---")

                            # Prepare full-length target tokens (labels shifted by 1)
                            labels_full = inputs.get("labels", None)
                            if labels_full is not None:
                                target_tokens_full = labels_full[0].detach().cpu()[1:]
                            else:
                                target_tokens_full = None

                            for pos in sel_pos_cpu.tolist():
                                if pos >= len(token_losses):
                                    continue  # skip invalid indices safely

                                harm_val = token_losses[pos].item()
                                pert_val = perturbed_token_losses[pos].item()
                                diff_val = diff_token_losses[pos].item()

                                if target_tokens_full is not None and pos < len(target_tokens_full):
                                    token_id = target_tokens_full[pos]
                                    token_text = tokenizer.decode([token_id], skip_special_tokens=False).replace(
                                        "▁", " "
                                    ).strip()
                                else:
                                    token_text = "<UNK>"

                                logger.info(
                                    f"pos {pos}: '{token_text}' | harmful {harm_val:.4f} | perturbed {pert_val:.4f} | diff {diff_val:.4f}"
                                )
                    except Exception as e:
                        logger.warning(f"Failed to log selected diff tokens: {e}")
                
                # Log the context around highest loss token for better understanding
                # Context logging around highest-loss token disabled per user request
                # if len(top_indices) > 0:
                #     highest_loss_idx = top_indices[0]
                #     context_window = 5
                #     start_idx = max(0, highest_loss_idx - context_window)
                #     end_idx = min(len(valid_input_tokens), highest_loss_idx + context_window + 1)
                #     
                #     context_input_tokens = valid_input_tokens[start_idx:end_idx]
                #     context_target_tokens = valid_target_tokens[start_idx:end_idx]
                #     context_losses = valid_losses[start_idx:end_idx]
                #     
                #     logger.info(f"--- CONTEXT AROUND HIGHEST LOSS TOKEN ---")
                #     for i, (inp_tok, tgt_tok, loss_val) in enumerate(zip(context_input_tokens, context_target_tokens, context_losses)):
                #         inp_text = tokenizer.decode([inp_tok], skip_special_tokens=False)
                #         tgt_text = tokenizer.decode([tgt_tok], skip_special_tokens=False)
                #         marker = " <<<< HIGHEST" if i == (highest_loss_idx - start_idx) else ""
                #         logger.info(f"    '{inp_text}' -> '{tgt_text}' (loss: {loss_val:.4f}){marker}")
            else:
                logger.info("No valid loss-contributing tokens found (all labels are -100)")
        else:
            logger.info("No labels found in inputs - cannot align with token losses")
            
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Error logging token-level losses at epoch {current_epoch}, step {step_in_epoch}: {e}")