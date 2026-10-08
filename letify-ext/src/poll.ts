/** Independent typed poll results and retained data, per spec "Editor extension". */
export async function poll<S extends { errors: Record<string, string> }, K extends keyof S & string>(
  state: S,
  command: K,
  read: (command: string) => Promise<unknown>,
  parse: (data: unknown) => S[K],
): Promise<void> {
  try {
    state[command] = parse(await read(command));
    delete state.errors[command];
  } catch (error) {
    state.errors[command] = error instanceof Error ? error.message : String(error);
  }
}
