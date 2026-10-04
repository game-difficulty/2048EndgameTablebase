// Task retention is shorter than replay retention. A missing remembered task
// must not make an otherwise available game look unavailable for analysis.
export async function recoverAnalysisJob(fetchJob, id) {
  try {
    return await fetchJob(id);
  } catch (error) {
    if (error.status === 404) return null;
    throw error;
  }
}
