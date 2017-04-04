%
% lstm_grad_check.m
%
% gradient check for compact LSTM code
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/04/2016
%
% didn't have time to make it more beautiful
%

dby_err = zeros(vocab_size, 1);
dWhy_err = zeros(vocab_size, hidden_size);
dU_err = zeros(hidden_size * 4, hidden_size);
dW_err = zeros(hidden_size * 4, vocab_size);

nby = zeros(vocab_size, 1);
nWhy = zeros(vocab_size, hidden_size);
nU = zeros(hidden_size * 4, hidden_size);
nW = zeros(hidden_size * 4, vocab_size);

increment = 1e-3;

%dby
for k=1:vocab_size
    delta = zeros(vocab_size, 1);
    delta(k) = increment;
    
    pre_loss = lstm_forward(xs, target, U, W, b, Why, by - delta, seq_length, h(:, 1), c(:, 1));
    post_loss = lstm_forward(xs, target, U, W, b, Why, by + delta, seq_length, h(:, 1), c(:, 1));
    
    numerical_grad = (post_loss - pre_loss) / (increment * 2);
    nby(k) = numerical_grad;
    analitic_grad = dby(k);
    dby_err(k) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
    
end

%dWhy
for k=1:vocab_size
    for kk=1:hidden_size
        
        delta = zeros(vocab_size, hidden_size);
        delta(k, kk) = increment;
        
        pre_loss = lstm_forward(xs, target, U, W, b, Why - delta, by, seq_length, h(:, 1), c(:, 1));
        post_loss = lstm_forward(xs, target, U, W, b, Why + delta, by, seq_length, h(:, 1), c(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nWhy(k, kk) = numerical_grad;
        analitic_grad = dWhy(k, kk);
        dWhy_err(k, kk) = abs(analitic_grad - numerical_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

% %dU
for k=1:(hidden_size*4)
    for kk=1:hidden_size
        
        delta = zeros(hidden_size*4, hidden_size);
        delta(k, kk) = increment;
        
        pre_loss = lstm_forward(xs, target, U - delta, W, b, Why, by, seq_length, h(:, 1), c(:, 1));
        post_loss = lstm_forward(xs, target, U + delta, W, b, Why, by, seq_length, h(:, 1), c(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nU(k, kk) = numerical_grad;
        analitic_grad = dU(k, kk);
        dU_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%dW
for k=1:(hidden_size*4)
    for kk=1:vocab_size
        
        delta = zeros(hidden_size*4, vocab_size);
        delta(k, kk) = increment;
        
        pre_loss = lstm_forward(xs, target, U, W - delta, b, Why, by, seq_length, h(:, 1), c(:, 1));
        post_loss = lstm_forward(xs, target, U, W + delta, b, Why, by, seq_length, h(:, 1), c(:, 1));
        
        numerical_grad = (post_loss - pre_loss) / (increment * 2);
        nW(k, kk) = numerical_grad;
        analitic_grad = dW(k, kk);
        dW_err(k, kk) = abs(numerical_grad - analitic_grad)/abs(numerical_grad + analitic_grad);
        
    end
end

%all errors should be relatively small - 1e-6 or less
fprintf('dby err = %.12f, max dby = %.12f, max nby = %.12f\n', max(dby_err(:)), max(dby(:)), max(nby(:)));
fprintf('dWhy err = %.12f, max dWhy = %.12f, max nWhy = %.12f\n', max(dWhy_err(:)), max(dWhy(:)), max(nWhy(:)));
fprintf('dU err = %.12f, max dU = %.12f, max nU = %.12f\n', max(dU_err(:)), max(dU(:)), max(nU(:)));
fprintf('dW err = %.12f, max dW = %.12f, max nW = %.12f\n', max(dW_err(:)), max(dW(:)), max(nW(:)));

