import { initializeApp } from 'firebase/app';
import {
  getAuth,
  GoogleAuthProvider,
  onAuthStateChanged,
  signInWithPopup,
  signOut,
  type User,
} from 'firebase/auth';
import firebaseConfig from '../../firebase-applet-config.json';

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const googleProvider = new GoogleAuthProvider();

// Store ID token strictly in memory (never localStorage)
let inMemoryToken: string | null = null;
let currentUser: User | null = null;
const listeners = new Set<(user: { email: string | null; uid: string } | null) => void>();

onAuthStateChanged(auth, async (user) => {
  currentUser = user;
  if (user) {
    try {
      inMemoryToken = await user.getIdToken();
    } catch {
      inMemoryToken = null;
    }
  } else {
    inMemoryToken = null;
  }
  const summary = user ? { email: user.email, uid: user.uid } : null;
  for (const cb of listeners) cb(summary);
});

export async function getToken(): Promise<string | null> {
  if (!currentUser) return null;
  inMemoryToken = await currentUser.getIdToken();
  return inMemoryToken;
}

export async function loginWithGoogle() {
  const cred = await signInWithPopup(auth, googleProvider);
  currentUser = cred.user;
  inMemoryToken = await cred.user.getIdToken();
  return { email: cred.user.email, uid: cred.user.uid };
}

export async function logout() {
  await signOut(auth);
  currentUser = null;
  inMemoryToken = null;
}

export function subscribeAuth(cb: (user: { email: string | null; uid: string } | null) => void) {
  listeners.add(cb);
  if (currentUser) {
    cb({ email: currentUser.email, uid: currentUser.uid });
  }
  return () => listeners.delete(cb);
}

declare global {
  interface Window {
    chromaAuth?: {
      getToken: typeof getToken;
      loginWithGoogle: typeof loginWithGoogle;
      logout: typeof logout;
      subscribeAuth: typeof subscribeAuth;
    };
  }
}

window.chromaAuth = {
  getToken,
  loginWithGoogle,
  logout,
  subscribeAuth,
};
