%
% lstm_forward.m
%
% lstm forward pass for grad check
%
% Author: Kamil Rocki <kmrocki@us.ibm.com>
% Created on: 02/04/2016
%

function loss = lstm_forward(xs, target, U, W, b, Why, by, seq_length, h_prev, c_prev)

	hidden_size = size(W, 1)/4;
	K = hidden_size;
	vocab_size = size(W, 2);

	h = zeros(hidden_size, seq_length);
	y = zeros(vocab_size, seq_length);
	probs = zeros(vocab_size, seq_length);

	h(:, 1) = h_prev;
	c(:, 1) = c_prev;

	loss = 0;

	for t = 2:seq_length

		%LSTM gates
        iofc(:, t) = W * xs(:, t) + U * h(:, t - 1) + b; 
        iofc(1:K*3, t) = sigmoid(iofc(1:K*3, t)); %iof - sigmoid
        iofc(K*3+1:end, t) = tanh(iofc(K*3+1:end, t)); %c - tanh

        c(:, t) = tanh( iofc(1:K, t) .* iofc(K*3+1:end, t) + ...
                            iofc(K*2+1:K*3, t) .* c(:, t - 1));
 
        h(:, t) = iofc(K*1+1:K*2, t) .* c(:, t);

		% update y
		y(:, t) = Why * h(:, t) + by;

		% compute probs
		probs(:, t) = exp(y(:, t)) ./ sum(exp(y(:, t)));

		% cross-entropy loss, sum logs of probabilities of target outputs
		loss = loss + sum(- log(probs(:, t)) .* target(:, t));

	end

end

