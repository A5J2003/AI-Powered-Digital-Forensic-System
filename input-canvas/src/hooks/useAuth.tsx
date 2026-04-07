import { useState, useEffect, createContext, useContext, ReactNode } from 'react';

// Mock types to match what the app expects
export interface User {
  id: string;
  email?: string;
}

export interface Session {
  access_token: string;
  user: User;
}

interface AuthContextType {
  user: User | null;
  session: Session | null;
  loading: boolean;
  signIn: (email: string, password: string) => Promise<{ error: Error | null }>;
  signUp: (email: string, password: string) => Promise<{ error: Error | null }>;
  signOut: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const AuthProvider = ({ children }: { children: ReactNode }) => {
  const [user, setUser] = useState<User | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check local storage for existing session
    const stored = localStorage.getItem('mock_auth_session');
    if (stored) {
      try {
        const parsedSession = JSON.parse(stored);
        if (parsedSession && parsedSession.user) {
          setSession(parsedSession);
          setUser(parsedSession.user);
        }
      } catch (e) {
        console.error("Failed to parse local session", e);
      }
    }
    setLoading(false);
  }, []);

  const signIn = async (email: string, _password: string) => {
    // In local dev mock, any email/password works
    const mockUser = { id: 'user_' + Date.now().toString(), email };
    const mockSession = { access_token: 'mock_token_' + Date.now().toString(), user: mockUser };
    
    setSession(mockSession);
    setUser(mockUser);
    localStorage.setItem('mock_auth_session', JSON.stringify(mockSession));
    
    return { error: null };
  };

  const signUp = async (email: string, password: string) => {
    // For local mock, signUp acts just like signIn (auto-login after creation)
    return signIn(email, password);
  };

  const signOut = async () => {
    setSession(null);
    setUser(null);
    localStorage.removeItem('mock_auth_session');
  };

  return (
    <AuthContext.Provider value={{ user, session, loading, signIn, signUp, signOut }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};
