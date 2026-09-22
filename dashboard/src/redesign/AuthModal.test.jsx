import { cleanup, render, screen, fireEvent, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { AuthModal } from './AuthModal';
import * as supabaseClient from '../lib/supabaseClient';

vi.mock('../lib/supabaseClient', () => ({
  signInWithPassword: vi.fn(),
  signUp: vi.fn(),
}));

describe('AuthModal', () => {
  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it('renders nothing when isOpen is false', () => {
    const { container } = render(<AuthModal isOpen={false} onClose={vi.fn()} />);
    expect(container.firstChild).toBeNull();
  });

  it('renders sign in view when isOpen is true', () => {
    render(<AuthModal isOpen={true} onClose={vi.fn()} />);
    expect(screen.getByRole('heading', { name: /sign in to nugget/i })).toBeInTheDocument();
    expect(screen.getByLabelText(/email address/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/^password$/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('switches between sign in and create account mode', () => {
    render(<AuthModal isOpen={true} onClose={vi.fn()} />);
    
    const createAccountBtn = screen.getByRole('button', { name: /create account/i });
    fireEvent.click(createAccountBtn);

    expect(screen.getByRole('heading', { name: /create your account/i })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign up/i })).toBeInTheDocument();

    const signInLink = screen.getByRole('button', { name: /sign in/i });
    fireEvent.click(signInLink);

    expect(screen.getByRole('heading', { name: /sign in to nugget/i })).toBeInTheDocument();
  });

  it('submits sign in and calls onSuccess', async () => {
    const onSuccess = vi.fn();
    const onClose = vi.fn();
    supabaseClient.signInWithPassword.mockResolvedValueOnce({
      user: { id: 'usr_1', email: 'test@example.com' },
      token: 'jwt.token.here',
    });

    render(<AuthModal isOpen={true} onClose={onClose} onSuccess={onSuccess} />);

    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: 'secret123' } });
    fireEvent.click(screen.getByRole('button', { name: /^sign in$/i }));

    await waitFor(() => {
      expect(supabaseClient.signInWithPassword).toHaveBeenCalledWith({
        email: 'test@example.com',
        password: 'secret123',
      });
      expect(onSuccess).toHaveBeenCalledWith(expect.objectContaining({
        user: { id: 'usr_1', email: 'test@example.com' },
      }));
      expect(onClose).toHaveBeenCalled();
    });
  });

  it('displays error message when sign in fails', async () => {
    supabaseClient.signInWithPassword.mockRejectedValueOnce(new Error('Invalid login credentials'));

    render(<AuthModal isOpen={true} onClose={vi.fn()} />);

    fireEvent.change(screen.getByLabelText(/email address/i), { target: { value: 'test@example.com' } });
    fireEvent.change(screen.getByLabelText(/^password$/i), { target: { value: 'wrongpass' } });
    fireEvent.click(screen.getByRole('button', { name: /^sign in$/i }));

    await waitFor(() => {
      expect(screen.getByText('Invalid login credentials')).toBeInTheDocument();
    });
  });
});
